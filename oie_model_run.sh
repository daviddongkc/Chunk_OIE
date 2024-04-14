python allennlp_run.py --cuda 0 --config config/oie_wiki_conll.json --epoch 5 --batch 32 --model trained_model/Chunk_OIE_wiki_conll_model
python allennlp_run.py --cuda 0 --config config/oie_sci_conll.json --epoch 5 --batch 32 --model trained_model/Chunk_OIE_sci_conll_model
python allennlp_run.py --cuda 0 --config config/oie_wiki_oia.json --epoch 5 --batch 32 --model trained_model/Chunk_OIE_wiki_oia_model
python allennlp_run.py --cuda 0 --config config/oie_sci_oia.json --epoch 5 --batch 32 --model trained_model/Chunk_OIE_sci_oia_model