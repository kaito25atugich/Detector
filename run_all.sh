# python generate_human_dataset.py
# python generate_ai_dataset.py
# python generate_ai_dataset.py --dataset_name writingprompts
# python detection.py --prompt --self_prompt # white-box detection
python detection.py --binoculars_only  --prompt --self_prompt # white-box detection
# python detection.py # black-box detection
# python detection.py --prompt # white-box detection
python detection.py --binoculars_only # black-box detection
python detection.py --binoculars_only  --prompt # white-box detection
# python detection.py --dataset_name writingprompts # black-box detection 
# python detection.py --dataset_name writingprompts --prompt # white-box detection
# python detection.py --dataset_name writingprompts --binoculars_only # black-box detection
# python detection.py --dataset_name writingprompts --binoculars_only  --prompt # white-box detection