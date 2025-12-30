import logging
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ...utils import check_dependency_available

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    Protocol = object  # type: ignore

logger = logging.getLogger(__name__)


class OCRModelProtocol(Protocol):
    def __init__(self, *args, **kwargs): ...


OCR_ENGINES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
SUPPORTED_ENGINES: Dict[str, List[Type[Any]]] = {}


def _ocr_model_class_by_name(model_name: str) -> Type[OCRModelProtocol]:
    from .deepseek_ocr import DeepSeekOCRModel
    from .got_ocr2 import GotOCR2Model
    from .hunyuan_ocr import HunyuanOCRModel
    from .paddleocr_vl import PaddleOCRVLModel

    mapping = {
        "DeepSeek-OCR": DeepSeekOCRModel,
        "GOT-OCR2_0": GotOCR2Model,
        "HunyuanOCR": HunyuanOCRModel,
        "PaddleOCR-VL": PaddleOCRVLModel,
    }
    if model_name not in mapping:
        raise ValueError(f"OCR model {model_name} not found.")
    return mapping[model_name]


def _check_format_with_engine(model_format: Optional[str], engine: str) -> bool:
    if model_format == "mlx":
        return engine == "mlx"
    return engine == "transformers"


def _default_ocr_spec(model_family: Any) -> Dict[str, Any]:
    return {
        "model_name": model_family.model_name,
        "model_format": "pytorch",
        "quantization": "none",
        "model_id": model_family.model_id,
        "model_hub": model_family.model_hub,
    }


def _get_ocr_specs(model_family: Any) -> List[Dict[str, Any]]:
    specs = getattr(model_family, "ocr_model_specs", None)
    filtered: List[Dict[str, Any]] = []
    if specs:
        for spec in specs:
            spec_hub = spec.get("model_hub", model_family.model_hub)
            if spec_hub != model_family.model_hub:
                continue
            filtered.append(spec.copy())
    default_spec = _default_ocr_spec(model_family)
    has_transformer = any(
        spec.get("model_format") in (None, "pytorch", "transformers")
        for spec in filtered
    )
    if not has_transformer:
        filtered.append(default_spec)
    return filtered


class TransformersOCREngine:
    engine_name = "transformers"

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        return check_dependency_available("transformers", "transformers")

    @classmethod
    def match(
        cls, model_family: Any, spec: Dict[str, Any], quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        model_format = spec.get("model_format")
        if model_format not in (None, "pytorch", "transformers"):
            return False, "Transformers OCR engine only supports pytorch format"
        return True


class MLXOCREngine:
    engine_name = "mlx"

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("mlx_vlm", "mlx_vlm")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match(
        cls, model_family: Any, spec: Dict[str, Any], quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_family.model_name != "DeepSeek-OCR":
            return False, "MLX OCR engine only supports DeepSeek-OCR for now"
        model_format = spec.get("model_format")
        if model_format != "mlx":
            return False, "MLX OCR engine only supports mlx format"
        if sys.platform != "darwin" or platform.processor() != "arm":
            return False, "MLX OCR engine only works on Apple silicon Macs"
        return True


def generate_engine_config_by_model_name(model_family: Any) -> None:
    model_name = model_family.model_name
    engines: Dict[str, List[Dict[str, Any]]] = OCR_ENGINES.get(model_name, {})
    specs = _get_ocr_specs(model_family)

    for spec in specs:
        model_format = spec.get("model_format")
        quantization = spec.get("quantization") or "none"
        for engine in SUPPORTED_ENGINES:
            if not _check_format_with_engine(model_format, engine):
                continue
            for engine_cls in SUPPORTED_ENGINES[engine]:
                if engine_cls.match(model_family, spec, quantization):
                    ocr_cls = _ocr_model_class_by_name(model_name)
                    engines.setdefault(engine, []).append(
                        {
                            "model_name": model_name,
                            "model_format": model_format,
                            "quantization": quantization,
                            "model_id": spec.get("model_id"),
                            "model_hub": spec.get("model_hub"),
                            "ocr_class": ocr_cls,
                        }
                    )
                    break

    OCR_ENGINES[model_name] = engines


def check_engine_by_model_name_and_engine(
    model_engine: str,
    model_name: str,
    model_format: Optional[str],
    quantization: Optional[str],
) -> Tuple[Type[OCRModelProtocol], Dict[str, Any]]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in OCR_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in OCR_ENGINES:
        raise ValueError(f"OCR model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in OCR_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = OCR_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name != param["model_name"]:
            continue
        if model_format and model_format != param["model_format"]:
            continue
        if quantization and quantization != param["quantization"]:
            continue
        return param["ocr_class"], param
    # fallback to first entry if quantization not specified
    if not model_format and not quantization and match_params:
        return match_params[0]["ocr_class"], match_params[0]
    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
