"""One-shot CLI: cross-sign a Matrix device using a user's SSSS recovery key.

Use case: an agent's Matrix device exists with valid Olm keys uploaded to the
homeserver, but isn't signed by the user's self-signing key. Peers running
clients that require cross-signed devices (Element by default) silently refuse
to share Megolm session keys with it, so the agent can't decrypt incoming
encrypted messages.

Manual fix is to log into Element as the user, unlock SSSS with the recovery
key, and verify the device. This script does the same crypto programmatically:

  1. Fetch the user's SSSS default key descriptor from account_data.
  2. Verify the supplied recovery key against the descriptor and derive the
     SSSS AES key.
  3. Decrypt the master / self-signing / user-signing private keys stored
     encrypted in account_data.
  4. Fetch the target device's published Olm/Megolm keys from /keys/query.
  5. Sign them with the self-signing private key.
  6. POST the signature to /keys/signatures/upload.

After this, peers see the device as cross-signed by the user's self-signing
key and will share Megolm keys with it.

Requirements:
    pip install 'mautrix[e2e]'

Usage:
    python tools/matrix-cross-sign.py \\
        --homeserver https://matrix.example.com \\
        --user-id @agent:example.com \\
        --access-token "$TOKEN" \\
        --device-id AGENT_DEVICE \\
        --recovery-key "EsT9 RzbW ..."

The user must also be verified once at the user level from a human's client
(Element → user profile → "Verify user") for the trust chain to close all the
way to that human — but this script makes the device side valid.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from mautrix.client import Client
from mautrix.crypto.cross_signing_key import CrossSigningSeeds
from mautrix.crypto.signature import sign_olm
from mautrix.crypto.ssss import Machine as SSSSMachine
from mautrix.types import DeviceID, EventType, KeyID, UserID


async def cross_sign_device(
    homeserver: str,
    user_id: str,
    access_token: str,
    device_id: str,
    recovery_key: str,
) -> None:
    client = Client(
        base_url=homeserver,
        mxid=UserID(user_id),
        token=access_token,
    )

    # 1-2: decrypt the SSSS default key using the recovery key
    ssss = SSSSMachine(client)
    key_id, key_metadata = await ssss.get_default_key_data()
    ssss_key = key_metadata.verify_recovery_key(key_id, recovery_key)

    # 3: fetch and decrypt the three cross-signing private seeds
    seeds = CrossSigningSeeds(
        master_key=await ssss.get_decrypted_account_data(
            EventType.CROSS_SIGNING_MASTER, ssss_key
        ),
        self_signing_key=await ssss.get_decrypted_account_data(
            EventType.CROSS_SIGNING_SELF_SIGNING, ssss_key
        ),
        user_signing_key=await ssss.get_decrypted_account_data(
            EventType.CROSS_SIGNING_USER_SIGNING, ssss_key
        ),
    )
    private_keys = seeds.to_keys()
    self_signing = private_keys.self_signing_key

    # 4: fetch the target device's published keys
    resp = await client.query_keys({UserID(user_id): [DeviceID(device_id)]})
    user_devices = resp.device_keys.get(UserID(user_id), {})
    device_keys = user_devices.get(DeviceID(device_id))
    if device_keys is None:
        raise SystemExit(
            f"device {device_id!r} not found for {user_id!r} — has it logged "
            "in and uploaded keys yet?"
        )

    # 5: sign with self-signing key and 6: upload
    signature = sign_olm(device_keys, self_signing)
    device_keys.signatures = {
        UserID(user_id): {KeyID.ed25519(self_signing.public_key): signature}
    }
    await client.upload_one_signature(UserID(user_id), DeviceID(device_id), device_keys)

    print(
        f"signed {user_id}:{device_id} with self-signing key "
        f"ed25519:{self_signing.public_key}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-sign a Matrix device using a user's SSSS recovery key."
    )
    parser.add_argument("--homeserver", required=True, help="https://homeserver URL")
    parser.add_argument("--user-id", required=True, help="@user:server")
    parser.add_argument("--access-token", required=True, help="user's access token")
    parser.add_argument("--device-id", required=True, help="device to sign")
    parser.add_argument("--recovery-key", required=True, help="SSSS recovery key")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )

    try:
        asyncio.run(
            cross_sign_device(
                homeserver=args.homeserver,
                user_id=args.user_id,
                access_token=args.access_token,
                device_id=args.device_id,
                recovery_key=args.recovery_key,
            )
        )
    except Exception as e:
        print(f"error: {e!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
