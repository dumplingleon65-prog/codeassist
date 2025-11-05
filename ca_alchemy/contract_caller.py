import json
import logging
import time

from dataclasses import dataclass

import requests

from eth_abi import encode, decode
from eth_keys import keys
from eth_utils import decode_hex, keccak, to_hex

ALCHEMY_BASE_URL = f"https://api.g.alchemy.com"
CONTRACT_ADDRESS = "0x0d3A2561883203a48E4227D41D37E9ffF81CAb85"
ALCHEMY_API_KEY = "wvs3CE89g2JwoshNNCMe1"
PAYMASTER_POLICY_ID = "a7563814-861e-49a1-b695-78a951ad0bc3"
ALCHEMY_CHAIN_ID = hex(685685)

logger = logging.getLogger(__name__)


@dataclass
class ContractArgs:
    accountAddress: str
    gitRef: str
    huggingFaceID: str


def encode_args(args: ContractArgs) -> str:
    selector = keccak(text="submitHFUpload(address,string,string)")[:4]
    args = encode(
        ["address", "string", "string"],
        [args.accountAddress, args.gitRef, args.huggingFaceID],
    )

    return to_hex(selector + args)


def sign(message_hash, private_key: str):
    logger.info(f"Signing message hash: {message_hash}")
    """Sign a hash directly with ECDSA."""
    priv_bytes = bytes.fromhex(private_key[2:])
    hash_bytes = decode_hex(message_hash)

    priv = keys.PrivateKey(priv_bytes)
    signature = priv.sign_msg_hash(hash_bytes)

    # eth_keys returns signature with v as 0/1, but Ethereum expects 27/28
    sig_bytes = signature.to_bytes()  # r + s + v (where v is 0 or 1)
    sig_bytes = sig_bytes[:-1] + bytes([sig_bytes[-1] + 27])  # Adjust v to 27/28

    return "0x" + sig_bytes.hex()


def call_get_account(owner_address: str) -> str:
    logger.info(f"Getting account for owner address: {owner_address}")
    d = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "wallet_requestAccount",
        "params": [{"signerAddress": owner_address}],
    }

    resp = requests.post(f"{ALCHEMY_BASE_URL}/v2/{ALCHEMY_API_KEY}", json=d)

    resp.raise_for_status()

    return resp.json().get("result").get("accountAddress")


def call_contract(args: ContractArgs):
    logger.info(f"Calling smart contract at {CONTRACT_ADDRESS}")
    args_hex = encode_args(args)
    prepare_calls_data = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "wallet_prepareCalls",
        "params": [
            {
                "capabilities": {
                    "paymasterService": {"policyId": PAYMASTER_POLICY_ID},
                    # "permissions": {
                    #    "context": f"0x01{user_key_data.get('deferredActionDigest')[2:]}"
                    # },
                },
                "calls": [
                    {
                        "to": CONTRACT_ADDRESS,
                        "data": args_hex,
                    }
                ],
                "from": args.accountAddress,
                "chainId": ALCHEMY_CHAIN_ID,
            }
        ],
    }

    resp = requests.post(
        f"{ALCHEMY_BASE_URL}/v2/{ALCHEMY_API_KEY}", json=prepare_calls_data
    )

    resp.raise_for_status()

    resp_json = resp.json()
    return resp_json


def sign_userop(data: dict, private_key: str) -> str:
    return sign(
        data.get("result").get("signatureRequest").get("rawPayload"), private_key
    )


def send_call(data: dict) -> dict:
    logger.info(f"Sending call to Alchemy API")
    resp = requests.post(f"{ALCHEMY_BASE_URL}/v2/{ALCHEMY_API_KEY}", json=data)

    resp.raise_for_status()

    return resp.json()


def get_call_status(call_ids: list[str]) -> dict:
    d = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "wallet_getCallsStatus",
        "params": call_ids,
    }

    resp = requests.post(f"{ALCHEMY_BASE_URL}/v2/{ALCHEMY_API_KEY}", json=d)

    resp.raise_for_status()

    return resp.json()


def submit_hf_upload(hf_id: str, git_hash: str):
    logger.info(f"Submitting Hugging Face upload for {hf_id} ({git_hash})...")
    user_key_map = {}

    with open("persistent-data/auth/userKeyMap.json", "r") as f:
        user_key_map = json.load(f)

    user_key_data = {}
    for key in user_key_map.keys():
        if "keys" not in user_key_map[key] or len(user_key_map[key].get("keys")) == 0:
            raise ValueError(f"No keys found for user. Did you sign in with Alchemy?")
        user_key_data = user_key_map[key].get("keys")[0]

    if len(user_key_data.keys()) == 0:
        raise ValueError(f"No keys found for user. Did you sign in with Alchemy?")

    # accountAddress is overloaded
    owner_address = user_key_data.get("publicKey")
    private_key = user_key_data.get("privateKey")
    accountAddress = call_get_account(owner_address)

    logger.info(f"Using account address: {accountAddress}")

    contract_args = {
        "huggingFaceID": hf_id,
        "accountAddress": accountAddress,
        "gitRef": git_hash,
    }

    resp_data = call_contract(ContractArgs(**contract_args))

    if "error" in resp_data:
        logger.info(f"Prepared call returned an error: {resp_data['error']}")

        with open("logs/smartcontract_error.json", "w") as f:
            json.dump(resp_data, f)

        return False

    signature = sign_userop(resp_data, private_key)

    call_data = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "wallet_sendPreparedCalls",
        "params": [
            {
                "type": resp_data["result"].get("type"),
                "data": resp_data["result"].get("data"),
                "chainId": resp_data["result"].get("chainId"),
                "signature": {"type": "secp256k1", "data": signature},
            }
        ],
    }

    sent_call = send_call(call_data)
    status = 100
    for i in range(10):
        resp = get_call_status(sent_call.get("result").get("preparedCallIds"))
        status = resp.get("result").get("status")
        if status == 200:
            logger.info("Call succeeded!")

            with open("logs/smartcontract_success.json", "w") as f:
                json.dump(resp, f)

            return True
        elif status != 100:
            logger.error(f"Call failed with status: {status}")

            with open("logs/smartcontract_error.json", "w") as f:
                json.dump(resp, f)

            return False
        logger.info(f"Call pending... (attempt {i + 1}/10)")
        time.sleep(5)
