# Installation
## Docker

- Ensure docker is installed

execute the following commands in the terminal
```bash
docker pull yeng2/interactivesequencetaxa
docker run -p 127.0.0.1:5006:5006/tcp yeng2/interactivesequencetaxa
```

Then, in your browser, go to `127.0.0.1:5006`


## From Source

- clone this repository `git clone https://github.com/ithetreehugger/InteractiveSequenceTaxa.git`
- ensure you have python3, pip3, and pipenv installed
- `cd InteractiveSequenceTaxa` 
- `tar -xvf lineage-data.tar.gz` 
- `pipenv install`
- `pipenv run bokeh serve --websocket-max-message-size 104857600 --show metagenomics --allow-websocket-origin=*`


# Using the Exapmle Data

To acces the example data, clone this git repository:
``` bash
git clone https://github.com/ithetreehugger/InteractiveSequenceTaxa.git
```

Then, the example data will be locate in ./InteractiveSequenceTaxa/ExampleData

There are 3 files:
- sequence-even-abridged.csv
- sequence-staggered-abridged.txt
- sequence-staggered.txt

The abridged versions are smaller samples, to improve loading time.

To load the example data in the application, click `choose File` and navigate to the sample data

# Using the Application

