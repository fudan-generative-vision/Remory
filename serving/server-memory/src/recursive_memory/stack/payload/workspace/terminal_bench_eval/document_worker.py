"""Private JSONL adapter around the unchanged JobBench document MCP servers.

Launched in JobBench's unprivileged Landlock sandbox, with only staged task
documents and the CPU document runtime available. No task/verifier tree mounts.
"""
import asyncio
import json
import os
from pathlib import Path
import sys


async def main():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    config = json.loads(sys.stdin.readline())
    params = StdioServerParameters(command=config["command"], args=config["args"],
                                  env=dict(os.environ), cwd=Path.cwd())
    async with stdio_client(params, errlog=sys.stderr) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            enabled = config.get("enabled_tools")
            tools = [t for t in tools if enabled is None or t.name in enabled]
            names = {t.name for t in tools}
            print(json.dumps({"ready": {"tools": [t.model_dump(mode="json", exclude_none=True)
                                                  for t in tools]}}), flush=True)
            while line := await asyncio.to_thread(sys.stdin.readline):
                try:
                    call = json.loads(line)
                    if call["name"] not in names:
                        raise ValueError("Unknown document tool")
                    result = await session.call_tool(call["name"], call.get("arguments", {}))
                    output = result.model_dump(mode="json", exclude_none=True)
                except Exception as exc:
                    output = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
                print(json.dumps({"result": output}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
