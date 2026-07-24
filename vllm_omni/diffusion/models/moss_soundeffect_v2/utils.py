from typing import Any


class PipelineUnit:
    def __init__(
        self,
        separate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ):
        self.separate_cfg = separate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(self, pipe: Any, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(
        self, unit: PipelineUnit, pipe: Any, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict
    ) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega
            )
        elif unit.separate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega
