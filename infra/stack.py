"""Everything the competition needs in AWS, as one CDK stack.

Storage
    One S3 bucket and one DynamoDB table; space/state.py describes what is in them.

Two networks
    locked   where runner tasks execute. It has no route to the internet. A
             runner can reach the egress proxy (NDIF, and read-only Hugging
             Face), the stack's own bucket, and the AWS endpoints Fargate needs
             to start a task. DNS is filtered to the same short list of names.
    open     where evaluator tasks execute, with ordinary internet access.

Three kinds of task
    runner      untrusted: a submission, the competition environment, the inputs
    evaluator   trusted: the scoring code and the private data
    proxy       always on: the locked network's only way out

One pipeline
    A Step Functions state machine, started by the Space once per submission:
    run, delete the submitter's keys, evaluate, record. It writes each outcome
    to the submission's item in the table.

The Space's identity
    An IAM user that can read and write the table and the bucket and start the
    pipeline, and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_route53resolver as resolver
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from constructs import Construct

from competition import Competition
from space.state import EVALUATING, FAILED, LINKS_VARIABLE, SCORED, SUBMISSION

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"  # assembled by `manage.py build`
PROXY_PORT = 3128
LINUX = ecs.RuntimePlatform(cpu_architecture=ecs.CpuArchitecture.X86_64, operating_system_family=ecs.OperatingSystemFamily.LINUX)


class Leaderboard(Stack):
    def __init__(self, scope: Construct, competition: Competition, **kwargs):
        super().__init__(scope, f"{competition.name}-leaderboard", **kwargs)
        self.competition = competition
        self.log_group = logs.LogGroup(self, "Logs", retention=logs.RetentionDays.SIX_MONTHS, removal_policy=RemovalPolicy.RETAIN)

        self._storage()
        self._networks()
        self._proxy()
        self._tasks()
        self._pipeline()
        self._space_user()

        for name, value in {
            "Table": self.table.table_name,
            "Bucket": self.bucket.bucket_name,
            "StateMachine": self.pipeline.state_machine_arn,
            "SpaceUserName": self.space_user.user_name,
            "LogGroup": self.log_group.log_group_name,
        }.items():
            cdk.CfnOutput(self, name, value=value)

    # --------------------------------------------------------------- storage --

    def _storage(self) -> None:
        # Both hold the record of the competition, so deleting the stack leaves them.
        self.bucket = s3.Bucket(
            self,
            "Files",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            # The pipeline deletes a submitter's keys when their run ends. This
            # is the backstop for a pipeline that never got that far.
            lifecycle_rules=[s3.LifecycleRule(prefix="credentials/", expiration=Duration.days(7))],
        )
        self.table = dynamodb.Table(
            self,
            "Submissions",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(point_in_time_recovery_enabled=True),
            removal_policy=RemovalPolicy.RETAIN,
        )

    # -------------------------------------------------------------- networks --

    def _networks(self) -> None:
        egress = self.competition.runner.egress
        self.namespace = f"{self.competition.name}.internal"

        # One availability zone each: a competition does not need more, and each
        # AWS endpoint below is billed per zone.
        self.locked = ec2.Vpc(
            self,
            "LockedNetwork",
            max_azs=1,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="proxy", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="runner", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24),
            ],
        )
        self.open = ec2.Vpc(
            self,
            "OpenNetwork",
            max_azs=1,
            nat_gateways=0,
            subnet_configuration=[ec2.SubnetConfiguration(name="evaluator", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)],
        )
        self.runner_subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)

        self.proxy_group = ec2.SecurityGroup(self, "ProxyGroup", vpc=self.locked, description="egress proxy")
        self.runner_group = ec2.SecurityGroup(self, "RunnerGroup", vpc=self.locked, allow_all_outbound=False, description="runner tasks")
        self.evaluator_group = ec2.SecurityGroup(self, "EvaluatorGroup", vpc=self.open, description="evaluator tasks")
        self.proxy_group.add_ingress_rule(self.runner_group, ec2.Port.tcp(PROXY_PORT), "runner tasks")
        self.runner_group.add_egress_rule(self.proxy_group, ec2.Port.tcp(PROXY_PORT), "the egress proxy")
        # The runner subnet routes only to this network and to S3, so "anywhere"
        # here can only ever mean the bucket and the AWS endpoints below.
        self.runner_group.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "S3 and AWS endpoints; there is no other route")

        # What Fargate needs to start a task with no internet: the image (ECR,
        # whose layers come from S3) and somewhere to send logs.
        this_account = {"StringEquals": {"aws:PrincipalAccount": self.account}}
        endpoint_group = ec2.SecurityGroup(self, "EndpointGroup", vpc=self.locked, allow_all_outbound=False, description="AWS endpoints")
        endpoint_group.add_ingress_rule(ec2.Peer.ipv4(self.locked.vpc_cidr_block), ec2.Port.tcp(443))
        for name, service in {
            "EcrApi": ec2.InterfaceVpcEndpointAwsService.ECR,
            "EcrDocker": ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
            "CloudWatchLogs": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
        }.items():
            endpoint = self.locked.add_interface_endpoint(name, service=service, subnets=self.runner_subnets, security_groups=[endpoint_group], open=False)
            # Only this account's own roles may use them. A submission carrying
            # someone else's AWS keys cannot ship data to that account's logs.
            endpoint.add_to_policy(iam.PolicyStatement(principals=[iam.AnyPrincipal()], actions=["*"], resources=["*"], conditions=this_account))

        s3_endpoint = self.locked.add_gateway_endpoint("S3", service=ec2.GatewayVpcEndpointAwsService.S3, subnets=[self.runner_subnets])
        s3_endpoint.add_to_policy(
            iam.PolicyStatement(
                principals=[iam.AnyPrincipal()],
                actions=["s3:GetObject", "s3:PutObject"],
                # The stack's bucket (through the task's links) and the bucket
                # ECR serves image layers from. No other bucket is reachable.
                resources=[self.bucket.arn_for_objects("*"), f"arn:aws:s3:::prod-{self.region}-starport-layer-bucket/*"],
            )
        )

        # DNS would otherwise be a way out: a lookup of <data>.attacker.example
        # reaches the attacker's name server. Only these names resolve.
        names = ["*.amazonaws.com", f"*.{self.namespace}"]
        for host in egress.allow + egress.read_only:
            names += [host, f"*.{host}"]
        allowed = resolver.CfnFirewallDomainList(self, "AllowedNames", domains=names)
        everything = resolver.CfnFirewallDomainList(self, "AllNames", domains=["*"])
        rules = resolver.CfnFirewallRuleGroup(
            self,
            "DnsRules",
            firewall_rules=[
                resolver.CfnFirewallRuleGroup.FirewallRuleProperty(action="ALLOW", firewall_domain_list_id=allowed.attr_id, priority=100),
                resolver.CfnFirewallRuleGroup.FirewallRuleProperty(action="BLOCK", block_response="NXDOMAIN", firewall_domain_list_id=everything.attr_id, priority=200),
            ],
        )
        resolver.CfnFirewallRuleGroupAssociation(self, "DnsRulesOnLockedNetwork", firewall_rule_group_id=rules.attr_id, vpc_id=self.locked.vpc_id, priority=101)

    # ----------------------------------------------------------------- proxy --

    def _proxy(self) -> None:
        egress = self.competition.runner.egress
        self.cluster = ecs.Cluster(self, "Cluster", vpc=self.locked)
        self.cluster.add_default_cloud_map_namespace(name=self.namespace)

        definition = ecs.FargateTaskDefinition(self, "Proxy", cpu=256, memory_limit_mib=512, runtime_platform=LINUX)
        definition.add_container(
            "proxy",
            image=ecs.ContainerImage.from_asset(str(ROOT / "infra" / "proxy"), platform=ecr_assets.Platform.LINUX_AMD64),
            environment={"EGRESS_ALLOW": ",".join(egress.allow), "EGRESS_READ_ONLY": ",".join(egress.read_only)},
            port_mappings=[ecs.PortMapping(container_port=PROXY_PORT)],
            logging=ecs.LogDrivers.aws_logs(stream_prefix="proxy", log_group=self.log_group),
        )
        ecs.FargateService(
            self,
            "ProxyService",
            cluster=self.cluster,
            task_definition=definition,
            desired_count=1,
            assign_public_ip=True,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_groups=[self.proxy_group],
            cloud_map_options=ecs.CloudMapOptions(name="proxy"),  # proxy.<name>.internal
            min_healthy_percent=100,
            # A proxy that cannot start fails the deploy in minutes, not hours.
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

    # ----------------------------------------------------------------- tasks --

    def _tasks(self) -> None:
        runner, evaluator = self.competition.runner, self.competition.evaluator

        # The runner's task role is created empty and stays empty: a submission
        # that asks the task for AWS credentials gets ones that can do nothing.
        self.runner_task = ecs.FargateTaskDefinition(self, "Runner", cpu=runner.cpu, memory_limit_mib=runner.memory, runtime_platform=LINUX)
        self.runner_container = self.runner_task.add_container(
            "runner",
            image=ecs.ContainerImage.from_asset(str(BUILD / "runner"), platform=ecr_assets.Platform.LINUX_AMD64),
            environment={
                "LEADERBOARD_PROXY": f"http://proxy.{self.namespace}:{PROXY_PORT}",
                # The one host reached without the proxy: the bucket the task's links point at.
                "LEADERBOARD_DIRECT": self.bucket.bucket_regional_domain_name,
            },
            logging=ecs.LogDrivers.aws_logs(stream_prefix="runner", log_group=self.log_group),
        )

        self.evaluator_task = ecs.FargateTaskDefinition(self, "Evaluator", cpu=evaluator.cpu, memory_limit_mib=evaluator.memory, runtime_platform=LINUX)
        self.evaluator_container = self.evaluator_task.add_container(
            "evaluator",
            image=ecs.ContainerImage.from_asset(str(BUILD / "evaluator"), platform=ecr_assets.Platform.LINUX_AMD64),
            secrets={name: ecs.Secret.from_secrets_manager(secretsmanager.Secret.from_secret_name_v2(self, f"Secret{name}", name)) for name in evaluator.secrets},
            logging=ecs.LogDrivers.aws_logs(stream_prefix="evaluator", log_group=self.log_group),
        )

    # -------------------------------------------------------------- pipeline --

    def _pipeline(self) -> None:
        """The state machine. Its input is what ``State._start`` in
        space/state.py sends: the submission id, the keys of the objects to
        read back, and for each task a link to the links it may use."""
        runner, evaluator = self.competition.runner, self.competition.evaluator
        item = {
            "pk": tasks.DynamoAttributeValue.from_string(SUBMISSION),
            "sk": tasks.DynamoAttributeValue.from_string(sfn.JsonPath.string_at("$.id")),
        }

        def record(step: str, status: str, **attributes: str) -> tasks.DynamoUpdateItem:
            """Set the submission's status, and any other attributes (given as
            JSON paths into the pipeline's state), on its item."""
            names = {"#status": "status"} | {f"#{name}": name for name in attributes}
            values = {":status": tasks.DynamoAttributeValue.from_string(status)}
            values |= {f":{name}": tasks.DynamoAttributeValue.from_string(text) for name, text in attributes.items()}
            return tasks.DynamoUpdateItem(
                self,
                step,
                table=self.table,
                key=item,
                update_expression="SET " + ", ".join(f"{name} = :{name[1:]}" for name in names),
                expression_attribute_names=names,
                expression_attribute_values=values,
                result_path=sfn.JsonPath.DISCARD,
            )

        def read(step: str, key: str, into: str) -> tasks.CallAwsService:
            """Read a JSON object a task uploaded: its text, to store, and its
            parsed form, to branch on."""
            return tasks.CallAwsService(
                self,
                step,
                service="s3",
                action="getObject",
                parameters={"Bucket": self.bucket.bucket_name, "Key": sfn.JsonPath.string_at(key)},
                iam_resources=[self.bucket.arn_for_objects("runs/*")],
                result_selector={"text": sfn.JsonPath.string_at("$.Body"), "parsed": sfn.JsonPath.string_to_json(sfn.JsonPath.string_at("$.Body"))},
                result_path=into,
            )

        def links(container: ecs.ContainerDefinition, task: str) -> list[tasks.ContainerOverride]:
            """Start a container with the one link the Space wrote for it."""
            variable = tasks.TaskEnvironmentVariable(name=LINKS_VARIABLE, value=sfn.JsonPath.string_at(f"$.{task}"))
            return [tasks.ContainerOverride(container_definition=container, environment=[variable])]

        run = tasks.EcsRunTask(
            self,
            "Run",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            cluster=self.cluster,
            task_definition=self.runner_task,
            launch_target=tasks.EcsFargateLaunchTarget(platform_version=ecs.FargatePlatformVersion.LATEST),
            subnets=self.runner_subnets,
            security_groups=[self.runner_group],
            container_overrides=links(self.runner_container, "runner"),
            task_timeout=sfn.Timeout.duration(Duration.seconds(runner.timeout_total + 600)),
            result_path=sfn.JsonPath.DISCARD,
        )
        delete_credentials = tasks.CallAwsService(
            self,
            "DeleteCredentials",
            service="s3",
            action="deleteObject",
            parameters={"Bucket": self.bucket.bucket_name, "Key": sfn.JsonPath.string_at("$.credentials_key")},
            iam_resources=[self.bucket.arn_for_objects("credentials/*")],
            result_path=sfn.JsonPath.DISCARD,
        )
        read_report = read("ReadReport", "$.report_key", "$.report")
        evaluate = tasks.EcsRunTask(
            self,
            "Evaluate",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            cluster=self.cluster,
            task_definition=self.evaluator_task,
            launch_target=tasks.EcsFargateLaunchTarget(platform_version=ecs.FargatePlatformVersion.LATEST),
            subnets=ec2.SubnetSelection(subnets=self.open.public_subnets),
            security_groups=[self.evaluator_group],
            assign_public_ip=True,
            container_overrides=links(self.evaluator_container, "evaluator"),
            task_timeout=sfn.Timeout.duration(Duration.seconds(evaluator.timeout + 600)),
            result_path=sfn.JsonPath.DISCARD,
        )
        read_results = read("ReadResults", "$.results_key", "$.results")

        # Three ways to end. The submission was scored; the submission was at
        # fault (its run failed, or its output could not be scored); or the
        # pipeline itself broke, which is recorded with `error` so that it does
        # not count against the team.
        scored = record("RecordScored", SCORED, results=sfn.JsonPath.string_at("$.results.text"))
        unusable = record("RecordUnusableOutput", FAILED, results=sfn.JsonPath.string_at("$.results.text"))
        run_failed = record("RecordFailedRun", FAILED, report=sfn.JsonPath.string_at("$.report.text"))
        broke = record("RecordPipelineFailure", FAILED, error=sfn.JsonPath.format("{}: {}", sfn.JsonPath.string_at("$.failure.Error"), sfn.JsonPath.string_at("$.failure.Cause")))
        broke.next(sfn.Fail(self, "PipelineFailed"))

        for step in (read_report, evaluate, read_results):
            step.add_catch(broke, result_path="$.failure")
        # Whatever happens to the run, the submitter's keys are deleted next.
        run.add_catch(delete_credentials, result_path="$.failure")

        definition = (
            run.next(delete_credentials)
            .next(
                sfn.Choice(self, "RunTaskCompleted?")
                .when(sfn.Condition.is_present("$.failure"), broke)
                .otherwise(
                    read_report.next(
                        sfn.Choice(self, "RunSucceeded?")
                        .when(sfn.Condition.boolean_equals("$.report.parsed.ok", False), run_failed)
                        .otherwise(
                            record("RecordEvaluating", EVALUATING, report=sfn.JsonPath.string_at("$.report.text"))
                            .next(evaluate)
                            .next(read_results)
                            .next(sfn.Choice(self, "OutputUsable?").when(sfn.Condition.boolean_equals("$.results.parsed.ok", True), scored).otherwise(unusable))
                        )
                    )
                )
            )
        )
        self.pipeline = sfn.StateMachine(
            self,
            "Pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=Duration.seconds(runner.timeout_total + evaluator.timeout + 3600),
        )

    # ------------------------------------------------------------ space user --

    def _space_user(self) -> None:
        # The user has no access key here: `manage.py deploy` mints one when it
        # publishes the Space and hands it over directly.
        self.space_user = iam.User(self, "SpaceUser")
        self.table.grant_read_write_data(self.space_user)
        self.bucket.grant_read_write(self.space_user)  # also what lets it sign the tasks' links
        self.pipeline.grant_start_execution(self.space_user)
