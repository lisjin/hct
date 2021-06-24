6/24: Modified `allennlp_models/structured_prediction/predictors/constituency_parser.py` by adding `predict_w_pos` to avoid double spacy tokenization.
    ```
    def predict_w_pos(self, sentence: str, pos: str) -> JsonDict:
        return self.predict_json({"sentence": sentence, "pos": pos})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        """
        if "pos" not in json_dict:
            spacy_tokens = self._tokenizer.tokenize(json_dict["sentence"])
            sentence_text = [token.text for token in spacy_tokens]
            pos_tags = [token.tag_ for token in spacy_tokens]
        else:
            sentence_text = json_dict["sentence"].split()
            pos_tags = json_dict["pos"].split()
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)
    ```
