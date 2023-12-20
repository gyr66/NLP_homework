python ./run.py \
--model_name_or_path gyr66/machine_translation \
--do_eval  \
--validation_file ./valid.json \
--do_predict \
--test_file ./test.json \
--output_dir ./eval \
--overwrite_output_dir \
--per_device_eval_batch_size 32 \
--predict_with_generate