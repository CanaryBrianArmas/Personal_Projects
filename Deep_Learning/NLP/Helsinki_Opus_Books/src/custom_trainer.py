# src/custom_trainer.py
from transformers import Trainer

class CustomSeq2SeqTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Sobrescribe el método prediction_step para llamar a model.generate y obtener
        las secuencias generadas durante la evaluación.
        """
        if not prediction_loss_only:
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_length=128  # Puedes parametrizar este valor si lo necesitas
            )
            return None, generated_tokens, None
        return None, None, None
