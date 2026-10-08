import os
import subprocess

script = """
if [ "$GIT_COMMITTER_EMAIL" = "khushking846@gmail.com" ]
then
    export GIT_COMMITTER_NAME="shikhar"
    export GIT_COMMITTER_EMAIL="vermashikhar380@gmail.com"
fi
if [ "$GIT_AUTHOR_EMAIL" = "khushking846@gmail.com" ]
then
    export GIT_AUTHOR_NAME="shikhar"
    export GIT_AUTHOR_EMAIL="vermashikhar380@gmail.com"
fi
"""

env = os.environ.copy()
env['FILTER_BRANCH_SQUELCH_WARNING'] = '1'

subprocess.run(["git", "filter-branch", "-f", "--env-filter", script, "--tag-name-filter", "cat", "--", "--branches", "--tags"], env=env)
