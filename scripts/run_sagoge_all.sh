####################### SAgoge #######################
# Convert dataset
# mox bucket-pangu-green-guiyang/yanghaitao/Datasets/Code/SAgoge/ in /cache/Data/SAgoge/raw/
bash run_sagoge.sh

# Select query seed
python -m seed_selection.main --config seed_selection/configs/default.yaml estimate
python -m seed_selection.main --config seed_selection/configs/default.yaml run
python -m seed_selection.main --config seed_selection/configs/default.yaml run --resume
python -m seed_selection.main --config seed_selection/configs/default.yaml analyze

