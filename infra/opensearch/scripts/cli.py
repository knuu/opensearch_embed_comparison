from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pystache
import typer
import yaml
from datasets import Dataset as HfDataset
from datasets import load_dataset
from opensearch_py_ml.ml_commons import MLCommonClient
from opensearchpy import OpenSearch, helpers
from pydantic import BaseModel, Field, FilePath, ValidationError

app = typer.Typer(no_args_is_help=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PathConfig:
    models_path: Path = _repo_root() / "config" / "models.yaml"
    opensearch_infra_config_path: Path = _repo_root() / "config" / "opensearch" / "infra.yaml"
    opensearch_app_config_path: Path = _repo_root() / "config" / "opensearch" / "app.yaml"


class ModelConfig(BaseModel):
    name: str = Field(min_length=1)
    embedding_dim: int = Field(gt=0)
    version: str = Field(min_length=1)
    model_format: Literal["TORCH_SCRIPT", "ONNX"] = Field()


class ModelsConfig:
    def __init__(self):
        self.models = self._load_models()

    def _load_models(self) -> dict[str, ModelConfig]:
        with PathConfig.models_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError("models.yaml の形式が不正です。")

        models: dict[str, ModelConfig] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("models.yaml の各エントリの形式が不正です。")
            try:
                model = ModelConfig.model_validate(item)
            except ValidationError as exc:
                raise ValueError("models.yaml のエントリが不正です。") from exc
            models[model.name] = model
        return models

    def __getitem__(self, model_name: str) -> ModelConfig:
        if model_name not in self.models:
            raise KeyError(f"モデル '{model_name}' が未定義です。")
        return self.models[model_name]


def load_model_config(model_name: str) -> ModelConfig:
    models_config = ModelsConfig()
    return models_config[model_name]


class HfDatasetConfig(BaseModel):
    name: str = Field(min_length=1)
    split: str = Field(min_length=1)


class JMTEBConfig(BaseModel):
    document: HfDatasetConfig
    query: HfDatasetConfig


class ColumnsConfig(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DatasetConfig(BaseModel):
    name: str = Field(min_length=1)
    type: Literal["JMTEB"] = Field()
    config: JMTEBConfig = Field()
    columns: ColumnsConfig = Field()

    @property
    def index_name(self) -> str:
        return f"index-{self.name}"


class DatasetsConfig:
    def __init__(self):
        self.datasets = self._load_datasets()

    def _load_datasets(self) -> dict[str, DatasetConfig]:
        config_path = _repo_root() / "config" / "datasets.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError("datasets.yaml の形式が不正です。")

        datasets: dict[str, DatasetConfig] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("datasets.yaml の各エントリの形式が不正です。")
            try:
                dataset = DatasetConfig.model_validate(item)
            except ValidationError as exc:
                raise ValueError("datasets.yaml のエントリが不正です。") from exc
            datasets[dataset.name] = dataset
        return datasets

    def __getitem__(self, dataset_name: str) -> DatasetConfig:
        if dataset_name not in self.datasets:
            raise KeyError(f"データセット '{dataset_name}' が未定義です。")
        return self.datasets[dataset_name]


def load_dataset_config(dataset_name: str) -> DatasetConfig:
    datasets_config = DatasetsConfig()
    return datasets_config[dataset_name]


class OpenSearchInfraConfig(BaseModel):
    host: str = Field(min_length=1)

    @classmethod
    def load(cls) -> "OpenSearchInfraConfig":
        with PathConfig.opensearch_infra_config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("infra.yaml の形式が不正です。")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError("host が未設定です。") from exc


class IndexTemplateConfig(BaseModel):
    name: str = Field(min_length=1)
    path: FilePath = Field()


class IngestPipelineConfig(BaseModel):
    name: str = Field(min_length=1)
    path: FilePath = Field()


class SearchPipelineConfig(BaseModel):
    name: str = Field(min_length=1)
    path: FilePath = Field()


class OpenSearchAppConfig(BaseModel):
    index_template: IndexTemplateConfig = Field()
    ingest_pipeline: IngestPipelineConfig = Field()
    search_pipeline: SearchPipelineConfig = Field()

    @classmethod
    def load(cls) -> "OpenSearchAppConfig":
        with PathConfig.opensearch_app_config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("app.yaml の形式が不正です。")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError("app.yaml の形式が不正です。") from exc


class OpenSearchClient:
    def __init__(self, config: OpenSearchInfraConfig):
        self.client = OpenSearch(hosts=[config.host])
        self.ml_client = MLCommonClient(self.client)

    def put_index_template(self, template_name: str, template_body: str) -> None:
        self.client.indices.put_index_template(
            name=template_name,
            body=json.loads(template_body),
        )

    def put_cluster_settings(self, settings: dict) -> None:
        self.client.cluster.put_settings(body=settings)

    def register_model(self, model_config: ModelConfig) -> str:
        """ML モデルを登録・デプロイして model_id を返す。"""
        return self.ml_client.register_pretrained_model(
            model_name=model_config.name,
            model_version=model_config.version,
            model_format=model_config.model_format,
            deploy_model=True,
        )

    def deregister_model(self, model_id: str) -> None:
        self.ml_client.undeploy_model(model_id=model_id)
        self.ml_client.delete_model(model_id=model_id)

    def register_ingest_pipeline(self, pipeline_id: str, pipeline_body: dict) -> None:
        self.client.ingest.put_pipeline(
            id=pipeline_id,
            body=pipeline_body,
        )

    def register_search_pipeline(self, pipeline_id: str, pipeline_body: dict) -> None:
        self.client.transport.perform_request(
            method="PUT",
            url=f"/_search/pipeline/{pipeline_id}",
            body=pipeline_body,
        )

    def create_index(self, index_name: str) -> None:
        self.client.indices.create(index=index_name)

    def search_model_group(self, request_body: dict) -> dict:
        return self.client.transport.perform_request(
            method="GET",
            url="/_plugins/_ml/model_groups/_search",
            body=request_body,
        )

    def search_model(self, request_body: dict) -> dict:
        return self.client.transport.perform_request(
            method="GET",
            url="/_plugins/_ml/models/_search",
            body=request_body,
        )


def put_index_template(
    client: OpenSearchClient,
    os_app_config: OpenSearchAppConfig,
    model_config: ModelConfig,
) -> None:
    with os_app_config.index_template.path.open("r", encoding="utf-8") as f:
        template = f.read()
    rendered_template = pystache.render(
        template,
        {
            "embedding_dim": model_config.embedding_dim,
            "ingest_pipeline_name": os_app_config.ingest_pipeline.name,
            "search_pipeline_name": os_app_config.search_pipeline.name,
        },
    )
    client.put_index_template(os_app_config.index_template.name, rendered_template)


def put_ingest_pipeline(
    client: OpenSearchClient,
    os_app_config: OpenSearchAppConfig,
    model_id: str,
) -> None:
    with os_app_config.ingest_pipeline.path.open("r", encoding="utf-8") as f:
        pipeline_body = json.load(f)
    rendered_pipeline = pystache.render(
        json.dumps(pipeline_body),
        {"model_id": model_id},
    )

    client.register_ingest_pipeline(
        os_app_config.ingest_pipeline.name, json.loads(rendered_pipeline)
    )


def put_search_pipeline(
    client: OpenSearchClient,
    os_app_config: OpenSearchAppConfig,
    model_id: str,
) -> None:
    with os_app_config.search_pipeline.path.open("r", encoding="utf-8") as f:
        pipeline_body = json.load(f)
    rendered_pipeline = pystache.render(
        json.dumps(pipeline_body),
        {"model_id": model_id},
    )

    client.register_search_pipeline(
        os_app_config.search_pipeline.name, json.loads(rendered_pipeline)
    )


@app.command("initialize-index-config")
def initialize_index_config(
    model: str = typer.Argument(..., help="モデル名（config/models.json のキー）"),
) -> None:
    """Mustache テンプレートをレンダリングして index template を登録する。"""

    model_config = load_model_config(model)
    os_app_config = OpenSearchAppConfig.load()
    os_infra_config = OpenSearchInfraConfig.load()

    os_client = OpenSearchClient(os_infra_config)

    initialize_opensearch_setting_for_ml(os_client)
    model_id = os_client.register_model(model_config)
    typer.secho(
        f"モデル '{model_config.name}' を登録・デプロイしました (model_id: {model_id})。",
        fg=typer.colors.GREEN,
    )
    put_ingest_pipeline(os_client, os_app_config, model_id)
    typer.secho(
        f"ingest pipeline '{os_app_config.ingest_pipeline.name}' を登録しました。",
        fg=typer.colors.GREEN,
    )
    put_search_pipeline(os_client, os_app_config, model_id)
    typer.secho(
        f"search pipeline '{os_app_config.search_pipeline.name}' を登録しました。",
        fg=typer.colors.GREEN,
    )
    put_index_template(
        os_client,
        os_app_config,
        model_config,
    )
    typer.secho(
        f"index template '{os_app_config.index_template.name}' を登録しました。",
        fg=typer.colors.GREEN,
    )


def initialize_opensearch_setting_for_ml(client: OpenSearchClient) -> None:
    client.put_cluster_settings(
        {
            "persistent": {
                "plugins.ml_commons.only_run_on_ml_node": "false",
                "plugins.ml_commons.model_access_control_enabled": "true",
                "plugins.ml_commons.native_memory_threshold": "99",
            }
        }
    )


def _bulk_dataset(
    os_client: OpenSearchClient,
    dataset_config: DatasetConfig,
    dataset: HfDataset,
    bulk_size: int,
) -> None:
    for i, raw_docs in enumerate(dataset.batch(batch_size=bulk_size)):
        documents = []
        for id_, text in zip(
            raw_docs[dataset_config.columns.id], raw_docs[dataset_config.columns.text]
        ):
            documents.append(
                {
                    "_index": dataset_config.index_name,
                    "_id": id_,
                    "text": text,
                }
            )
        helpers.bulk(os_client.client, documents, max_retries=3, request_timeout=600)
        print(f"{(i + 1) * bulk_size} 件まで登録しました。")


@app.command("bulk-dataset")
def bulk_dataset(
    dataset: str = typer.Argument(..., help="データセット名"),
    bulk_size: int = typer.Option(1000, help="一括登録するドキュメント数"),
) -> None:
    """データセットを一括登録する。"""
    dataset_config = load_dataset_config(dataset)
    os_infra_config = OpenSearchInfraConfig.load()
    os_client = OpenSearchClient(os_infra_config)
    os_client.create_index(dataset_config.index_name)
    typer.secho(
        f"インデックス '{dataset_config.index_name}' を作成しました。",
        fg=typer.colors.GREEN,
    )

    typer.secho(f"データセット '{dataset}' をロードします。", fg=typer.colors.GREEN)
    dataset = load_dataset(
        "sbintuitions/JMTEB",
        name=dataset_config.config.document.name,
        split=dataset_config.config.document.split,
        trust_remote_code=True,
    )
    typer.secho(f"ドキュメント数: {len(dataset)}", fg=typer.colors.GREEN)

    typer.secho("ドキュメントを一括登録します。", fg=typer.colors.GREEN)
    _bulk_dataset(os_client, dataset_config, dataset, bulk_size)
    typer.secho("ドキュメントの一括登録が完了しました。", fg=typer.colors.GREEN)


@app.command("deregister-models")
def deregister_models() -> None:
    """登録されているすべてのモデルを登録解除する。"""
    os_infra_config = OpenSearchInfraConfig.load()
    os_client = OpenSearchClient(os_infra_config)

    response = os_client.search_model_group({"query": {"match_all": {}}})
    model_group_ids = [
        item.get("_id") for item in response.get("hits", {}).get("hits", []) if item.get("_id")
    ]
    for model_group_id in model_group_ids:
        response = os_client.search_model({"query": {"term": {"model_group_id": model_group_id}}})
        model_ids = [
            item.get("_id") for item in response.get("hits", {}).get("hits", []) if item.get("_id")
        ]
        for model_id in model_ids:
            os_client.deregister_model(model_id=model_id)
            print(f"モデル '{model_id}' を登録解除しました。")


if __name__ == "__main__":
    app()
