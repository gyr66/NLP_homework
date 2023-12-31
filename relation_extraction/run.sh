python run_classification.py  --model_name_or_path bert-base-uncased \
 --dataset_name sem_eval_2010_task_8 \
 --label_column_name relation \
 --output_dir ./output/relation_extraction_bert_base_uncased \
 --num_train_epochs 10 \
 --per_device_train_batch_size 64 \
 --per_device_eval_batch_size 64 \
 --metric_name f1 \
 --logging_strategy epoch \
 --save_strategy epoch \
 --evaluation_strategy epoch \
 --load_best_model_at_end \
 --metric_for_best_model f1 \
 --do_train --do_eval --do_predict \
 --push_to_hub