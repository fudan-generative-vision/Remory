"""Own a native app-server for intercepted local REMORY commands."""
from contextlib import contextmanager
import os
import subprocess

from .transport import RPC, connect, endpoint_path, write_json


@contextmanager
def host(session, binary, workspace, env, thread_id, model, unsafe=False):
    socket = endpoint_path(session)
    endpoint = 'unix://' + str(socket)
    child = rpc = None
    with (session / 'app-server.stderr.log').open('a') as log:
        try:
            os.chmod(log.name, 0o600)
            child = subprocess.Popen([str(binary), 'app-server', '--listen', endpoint],
                cwd=workspace, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True)
            rpc = RPC(connect(socket, child), session)
            rpc.call('initialize', dict(clientInfo=dict(name='remory-interactive', version='2'),
                                       capabilities=dict(experimentalApi=True)))
            rpc.send(dict(method='initialized'))
            options = dict(cwd=str(workspace), model=model, modelProvider='remory_local',
                           approvalPolicy='never' if unsafe else 'on-request',
                           sandbox='danger-full-access' if unsafe else 'workspace-write')
            if thread_id:
                rpc.root_id = thread_id
                response = rpc.call('thread/resume', dict(options, threadId=thread_id, excludeTurns=True))
            else:
                response = rpc.call('thread/start', dict(options, ephemeral=False))
                # Native 0.149 defers writing a new rollout until a durable
                # operation. The TUI's resume bootstrap needs that file even
                # while the thread is still loaded by this app-server. Naming
                # it persists the empty thread without creating a turn or goal.
                rpc.call('thread/name/set', dict(threadId=response['thread']['id'], name=workspace.name))
            rpc.root_id = thread_id = response['thread']['id']
            goal = rpc.call('thread/goal/get', dict(threadId=thread_id))
            write_json(session / 'goal.json', goal)
            info = dict(endpoint=endpoint, thread_id=thread_id, pid=os.getpid(), child_pid=child.pid,
                        codex=str(binary), workspace=str(workspace))
            write_json(session / 'interactive.json', info)
            yield info
        finally:
            if rpc:
                rpc.close()
            if child and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
