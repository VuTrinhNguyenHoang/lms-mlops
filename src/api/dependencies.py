from prefect.deployments import run_deployment

async def trigger_deployment(name: str, parameters: dict):
    flow_run = await run_deployment(
        name=name,
        parameters=parameters,
        timeout=0,
    )
    return str(flow_run.id)
