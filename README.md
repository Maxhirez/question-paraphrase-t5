# question-paraphrase-t5
Support scripts for [question-paraphrase-t5](https://huggingface.co/maxhirez/question-paraphrase-t5) model

*(From model card):*
LoRA fine-tune of Google-T5-base for generative question paraphrasing

## Model Details

### Model Description

Originally develped as a proof-of-concept mechanism for preventing the sharing of knowledge check questions and answers in e-learning, generating on-the-fly paraphrasing of knowledge check questions so no two learners would see the same thing and could not effectively share questions/answers with others.  The model was set aside.  Its effectiveness for the task was never fully evaluated, though certain shortcomings were apparent in some circusmtances.


## Bias, Risks, and Limitations

Noted tendency to switch POV (ie, "How do you do ABC?"->"How do I do ABC?") and in the case of questions requiring a high degree of regime expertise, paraphrases tended not to vary significantly from original source.

## Manifest
&bull; <ins>requirements.txt:</ins> conda environment requirements.

&bull; <ins>t5-lora-inference.py:</ins> Model usage demo.  Asks user for questions and generates paraphrasings of those questions until user types "exit" break word. Uses Apple Metal Performance Shaders with fallback to CPU.

&bull; <ins>t5-lora-paraphrase-mps.py:</ins> Training script.  Downloads BERT T5 model and LoRA finetunes against Glue MRPC and QQP datasets. Uses Apple Metal Performance Shaders with fallback to CPU.
