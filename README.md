# NeuralNine RNN

<i><b>DISCLAIMER:</b> This is not investing advice. I am not a professional who is qualified in giving any financial advice. This is a Python program purely about coding with financial data from the U.S. stock market.</i>

This algorithm predicts stock prices in Python using recurrent neural networks and machine learning. Practice run from YouTuber <a href="https://www.youtube.com/watch?v=PuZY9q-aKLw">NeuralNine</a>. <b>[Published on February 2nd, 2021]</b>

## Quickstart Install Instructions

It is highly recommended to run this program in a ``conda`` environment. You can find further instructions on how to install ```miniconda.sh``` on your computer based on its operating system <a href="https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions"><u>here</u>.</a>

Now, download and install ```miniconda.sh``` by following these steps:

1. Run the following four commands to download and install the latest macOS installer for Apple Silicon. Line by line, these commands:

* Create a new folder named `miniconda3` in your home directory.
* Download the macOS Miniconda installation script for your Apple Silicon architecture and save the script as ```miniconda.sh``` in the miniconda3 folder.
* Run the ```miniconda.sh``` installation script in silent mode using ``Bash``.
* Remove the ```miniconda.sh``` installation script file after installation is complete.

```bash
$ mkdir -p ~/miniconda3
$ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
```

1. After installing, close and reopen your terminal application or refresh it by running the following command:

```bash
$ source ~/miniconda3/bin/activate
```

3. Then, initialize conda on all available shells by running the following command:

```bash
$ conda init --all
```

### Virtual Environment

Now that has already been set up, you can create a ``conda`` environment and set up its interpreter as ``Python 3.10``:

```bash
$ conda create -n neural_nine python=3.10
```

Now you have set up an isolated environment called ``neural_nine``, a sandbox-like structure to install everything mentioned in the `requirements.txt` file. Then you should activate the ``conda`` environment by using the command:

```bash
$ conda activate neural_nine
```

Next, you must absolutely make sure your Python is compiled for ``ARM64`` when creating ``conda`` or it will not work:

```bash
(neural_nine) $ conda config --env --set subdir osx-arm64
```

To deactivate the ``conda`` environment:

```bash
(neural_nine) $ conda deactivate
```

### Requirements

Please install the required ``pip`` packages by running the following commands:

```bash
(neural_nine) $ pip install --upgrade pip setuptools wheel
```

```bash
(neural_nine) $ pip install -r requirements.txt
```
