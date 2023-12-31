from model import BertForRelationExtraction

BertForRelationExtraction.register_for_auto_class("AutoModelForSequenceClassification")
model = BertForRelationExtraction.from_pretrained(
    "gyr66/relation_extraction_bert_base_uncased"
)
model.push_to_hub("gyr66/relation_extraction_bert_base_uncased")
