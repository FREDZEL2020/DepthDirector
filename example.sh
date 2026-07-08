python tools/recam/inference.py --video_path assets/examples/anne.mp4 --out_dir assets/examples/preprocess/anne --video_length 81 --depth_model Pi3X --task process

python tools/recam/captioner.py --root_path assets/examples/preprocess/anne \
    --task process --dataset test_all

bash scripts/test.sh configs/depthdirector.yaml step-10000 assets/examples/preprocess/anne/metadata_test_all.csv