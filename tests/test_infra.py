"""The synthesized stack. These pin down the properties the design depends on:
what the untrusted runner task can reach and what it is allowed to do."""

import json

import pytest

import competition as comp
import manage
from space.state import LINKS_VARIABLE

cdk = pytest.importorskip("aws_cdk")


@pytest.fixture(scope="module")
def template() -> dict:
    from infra.stack import Leaderboard

    competition = comp.load()
    competition.evaluator.secrets = ["JUDGE_API_KEY"]
    manage.build(competition)  # the image assets are read from build/
    app = cdk.App()
    stack = Leaderboard(app, competition, env=cdk.Environment(account="123456789012", region="us-east-1"))
    return app.synth().get_stack_by_name(stack.stack_name).template


def of_type(template: dict, kind: str) -> dict:
    return {name: r for name, r in template["Resources"].items() if r["Type"] == kind}


def one(template: dict, kind: str, prefix: str) -> tuple[str, dict]:
    found = [(name, r) for name, r in of_type(template, kind).items() if name.startswith(prefix)]
    assert len(found) == 1, [name for name, _ in found]
    return found[0]


def definition(template: dict) -> dict:
    _, machine = one(template, "AWS::StepFunctions::StateMachine", "Pipeline")
    parts = machine["Properties"]["DefinitionString"]["Fn::Join"][1]
    return json.loads("".join(part if isinstance(part, str) else "ref" for part in parts))


def test_runner_task_role_can_do_nothing(template):
    role, _ = one(template, "AWS::IAM::Role", "RunnerTaskRole")
    attached = [p for p in of_type(template, "AWS::IAM::Policy").values() if {"Ref": role} in p["Properties"].get("Roles", [])]
    assert attached == []
    assert "ManagedPolicyArns" not in template["Resources"][role]["Properties"]


def test_runner_container_gets_no_secrets_and_the_evaluator_does(template):
    _, runner = one(template, "AWS::ECS::TaskDefinition", "Runner")
    _, evaluator = one(template, "AWS::ECS::TaskDefinition", "Evaluator")
    runner_container, evaluator_container = runner["Properties"]["ContainerDefinitions"][0], evaluator["Properties"]["ContainerDefinitions"][0]
    assert "Secrets" not in runner_container
    assert [s["Name"] for s in evaluator_container["Secrets"]] == ["JUDGE_API_KEY"]
    assert runner_container["Image"] != evaluator_container["Image"]
    assert {e["Name"] for e in runner_container["Environment"]} == {"LEADERBOARD_PROXY", "LEADERBOARD_DIRECT"}


def test_runner_may_only_talk_to_the_proxy_and_aws_endpoints(template):
    group, runner = one(template, "AWS::EC2::SecurityGroup", "RunnerGroup")
    inline = [(rule["FromPort"], rule.get("CidrIp")) for rule in runner["Properties"].get("SecurityGroupEgress", [])]
    to_proxy = [r["Properties"] for r in of_type(template, "AWS::EC2::SecurityGroupEgress").values() if r["Properties"]["GroupId"]["Fn::GetAtt"][0] == group]
    assert inline == [(443, "0.0.0.0/0")]
    assert [(r["FromPort"], r["DestinationSecurityGroupId"]["Fn::GetAtt"][0].startswith("ProxyGroup")) for r in to_proxy] == [(3128, True)]
    assert "SecurityGroupIngress" not in runner["Properties"]


def test_runner_subnet_has_no_route_to_the_internet(template):
    routes = of_type(template, "AWS::EC2::Route")
    assert routes and all("proxy" in name or "evaluator" in name for name in routes), list(routes)
    assert not of_type(template, "AWS::EC2::NatGateway")


def test_s3_is_reachable_only_for_the_stack_bucket_and_image_layers(template):
    gateway = [e for e in of_type(template, "AWS::EC2::VPCEndpoint").values() if e["Properties"]["VpcEndpointType"] == "Gateway"]
    (statement,) = gateway[0]["Properties"]["PolicyDocument"]["Statement"]
    assert statement["Action"] == ["s3:GetObject", "s3:PutObject"]
    assert len(statement["Resource"]) == 2 and "starport-layer-bucket" in json.dumps(statement["Resource"][1])


def test_interface_endpoints_serve_only_this_account(template):
    interfaces = [e for e in of_type(template, "AWS::EC2::VPCEndpoint").values() if e["Properties"]["VpcEndpointType"] == "Interface"]
    assert len(interfaces) == 3
    for endpoint in interfaces:
        (statement,) = endpoint["Properties"]["PolicyDocument"]["Statement"]
        assert statement["Condition"] == {"StringEquals": {"aws:PrincipalAccount": "123456789012"}}


def test_dns_resolves_only_allowed_names_on_the_locked_network(template):
    lists = {name: r["Properties"]["Domains"] for name, r in of_type(template, "AWS::Route53Resolver::FirewallDomainList").items()}
    allowed = next(domains for name, domains in lists.items() if name.startswith("AllowedNames"))
    assert set(allowed) == {"*.amazonaws.com", "*.looped-lens.internal", "api.ndif.us", "*.api.ndif.us", "huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co"}
    _, group = one(template, "AWS::Route53Resolver::FirewallRuleGroup", "DnsRules")
    assert [(r["Action"], r["Priority"]) for r in group["Properties"]["FirewallRules"]] == [("ALLOW", 100), ("BLOCK", 200)]
    _, association = one(template, "AWS::Route53Resolver::FirewallRuleGroupAssociation", "DnsRules")
    assert association["Properties"]["VpcId"]["Ref"].startswith("LockedNetwork")


def test_pipeline_hands_each_task_one_link(template):
    states = definition(template)["States"]
    run, evaluate = states["Run"]["Parameters"], states["Evaluate"]["Parameters"]
    handed = lambda task: task["Overrides"]["ContainerOverrides"][0]["Environment"]
    assert handed(run) == [{"Name": LINKS_VARIABLE, "Value.$": "$.runner"}]
    assert handed(evaluate) == [{"Name": LINKS_VARIABLE, "Value.$": "$.evaluator"}]
    assert "AssignPublicIp" not in run["NetworkConfiguration"]["AwsvpcConfiguration"]
    assert evaluate["NetworkConfiguration"]["AwsvpcConfiguration"]["AssignPublicIp"] == "ENABLED"


def test_pipeline_always_deletes_credentials_after_the_run(template):
    states = definition(template)["States"]
    assert states["Run"]["Next"] == "DeleteCredentials" and states["Run"]["Catch"][0]["Next"] == "DeleteCredentials"
    assert states["DeleteCredentials"]["Resource"].endswith("aws-sdk:s3:deleteObject")
    written = {name: state["Parameters"]["ExpressionAttributeValues"][":status"]["S"] for name, state in states.items() if name.startswith("Record")}
    assert written == {"RecordEvaluating": "evaluating", "RecordScored": "scored", "RecordUnusableOutput": "failed", "RecordFailedRun": "failed", "RecordPipelineFailure": "failed"}
    assert ":error" in states["RecordPipelineFailure"]["Parameters"]["ExpressionAttributeValues"]  # what keeps it off the rate limit


def test_space_user_is_limited_to_its_three_jobs(template):
    user, _ = one(template, "AWS::IAM::User", "SpaceUser")
    policies = [p for p in of_type(template, "AWS::IAM::Policy").values() if {"Ref": user} in p["Properties"].get("Users", [])]
    actions = set()
    for policy in policies:
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
            actions |= set([statement["Action"]] if isinstance(statement["Action"], str) else statement["Action"])
    assert {a.split(":")[0] for a in actions} == {"dynamodb", "s3", "states"}
    assert {a for a in actions if a.startswith("states:")} == {"states:StartExecution"}
    # Its key is minted at publish time and lives only in the Space.
    assert not of_type(template, "AWS::IAM::AccessKey") and not of_type(template, "AWS::SecretsManager::Secret")
