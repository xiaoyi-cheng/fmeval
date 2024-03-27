from typing import Optional

from langchain.chat_models import BedrockChat
from langchain.embeddings import BedrockEmbeddings

from fmeval.model_runners.util import get_bedrock_runtime_client


BEDROCK_MODEL_ID_DEFAULT = "anthropic.claude-v2"
BEDROCK_MODEL_ID_EMBEDDINGS = "amazon.titan-embed-text-v1"


def get_bedrock_model(model_id: Optional[str] = BEDROCK_MODEL_ID_DEFAULT) -> BedrockChat:
    client = get_bedrock_runtime_client()
    return BedrockChat(
        model_id=model_id,
        model_kwargs={"temperature": 0.1},
        client=client
    )


def get_bedrock_embedding(model_id: Optional[str] = BEDROCK_MODEL_ID_EMBEDDINGS) -> BedrockEmbeddings:
    client = get_bedrock_runtime_client()
    return BedrockEmbeddings(
        model_id=model_id ,
        client=client,
    )


"""

bedrock_model = BedrockChat(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
    endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
    model_id=config["model_id"],
    model_kwargs=config["model_kwargs"],
)

# init the embeddings
bedrock_embeddings = BedrockEmbeddings(
    credentials_profile_name=config["credentials_profile_name"],
    region_name=config["region_name"],
)

"""
