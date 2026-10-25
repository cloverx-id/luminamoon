from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def prune_deployments(
    repo: str,
    token: str,
    environment: str = "github-pages",
    retention: int = 3,
) -> None:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "luminamoon-ci",
    }
    url = f"https://api.github.com/repos/{repo}/deployments?environment={environment}&per_page=100"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            deployments = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Failed to fetch deployments for {environment}: {e}", file=sys.stderr)
        return

    deployments.sort(key=lambda d: d.get("id", 0), reverse=True)
    to_prune = deployments[retention:]
    print(
        f"Found {len(deployments)} deployments for '{environment}'. "
        f"Retaining latest {retention}, pruning {len(to_prune)} older entries."
    )

    for dep in to_prune:
        dep_id = dep["id"]
        # 1. Set status to inactive
        status_url = f"https://api.github.com/repos/{repo}/deployments/{dep_id}/statuses"
        try:
            req_st = urllib.request.Request(
                status_url,
                data=json.dumps({"state": "inactive"}).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(req_st)
            print(f"Deployment {dep_id} set to inactive.")
        except urllib.error.HTTPError as e:
            print(f"Warning setting {dep_id} inactive: {e.code} {e.reason}", file=sys.stderr)

        # 2. Delete deployment record
        del_url = f"https://api.github.com/repos/{repo}/deployments/{dep_id}"
        try:
            req_del = urllib.request.Request(del_url, headers=headers, method="DELETE")
            with urllib.request.urlopen(req_del) as del_resp:
                print(f"Deleted deployment {dep_id} (HTTP {del_resp.status})")
        except urllib.error.HTTPError as e:
            print(f"Warning deleting deployment {dep_id}: {e.code} {e.reason}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune old GitHub deployments to enforce retention limits."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GH_REPO")
        or os.environ.get("GITHUB_REPOSITORY")
        or "cloverx-id/luminamoon",
    )
    parser.add_argument("--environment", default="github-pages")
    parser.add_argument("--retention", type=int, default=3)
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "Warning: GH_TOKEN or GITHUB_TOKEN not set. Skipping deployment pruning.",
            file=sys.stderr,
        )
        return

    prune_deployments(
        repo=args.repo,
        token=token,
        environment=args.environment,
        retention=args.retention,
    )


if __name__ == "__main__":
    main()
