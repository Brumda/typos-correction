<h1 align="center">
<p>Robust neural embeddings for typos correction
</h1>

### These files are for my bachelor's thesis.  

All the python files in ./implementation were used on MetaCentrum clusters via the shell scripts in ./metacentrum folder, 
and each has its corresponding .yaml file to set up the environment needed to run the program.  
The jupyter notebooks were used for data manipulation.
- ./data_extraction.ipynb was used to split up the dataset and create all the files in /data folder
- ./data_extraction.ipynb was used to create all the graphs
- ./model_errors.ipynb was used to find examples of the errors the models made

All the evaluated data are uploaded to [WandB](https://wandb.ai/martin-elias-ctu-fit/Benchmarks)
and processed via API call, but the raw data are also available in the ./benchmark_results folder   

All the fine-tuned models are available on my [Hugging Face profile](https://huggingface.co/brumda)

Some of the NeuSpell toolkit's files were modified to fit my use case or to fix issues that the library had and are
available [here](https://github.com/Brumda/neuspell-fixed)