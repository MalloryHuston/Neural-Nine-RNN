# NeuralNine RNN

This algorithm predicts stock prices in Python using recurrent neural networks and machine learning. Practice run from YouTuber <a href="https://www.youtube.com/watch?v=PuZY9q-aKLw">NeuralNine</a>. <b>[Published on February 2nd, 2021]</b>

## Quickstart Install Instructions

It is highly recommended to run this program in a ``conda`` environment. You can find further instructions on how to install ``miniconda.sh`` on your computer if you have a macOS that is an Apple Silicon by clicking <a href="https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2">this link here.</a> Download and install ``miniconda.sh`` by following these steps:

1. Run the following four commands to download and install the latest macOS installer for Apple Silicon. Line by line, these commands:

* Create a new directory named “miniconda3” in your home directory.
* Download the macOS Miniconda installation script for your Apple Silicon architecture and save the script as ``miniconda.sh`` in the miniconda3 directory.
* Run the ``miniconda.sh`` installation script in silent mode using bash.
* Remove the ``miniconda.sh`` installation script file after installation is complete.

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

2. After installing, close and reopen your terminal application or refresh it by running the following command:

```bash
source ~/miniconda3/bin/activate
```

3. Then, initialize conda on all available shells by running the following command:

```bash
conda init --all
```

### Virtual Environment

Now that has already been set up, you can create a ``conda`` environment and set up its interpreter as ``Python 3.10``:

<pre>
<code>$ conda create -n neural_nine python=3.10</code>
</pre>

Now you have set up an isolated environment called ``neural_nine``, a sandbox-like structure to install everything mentioned in the ``requirements.txt`` file. Then you should activate the ``conda`` environment by using the command:

<pre>
<code>$ conda activate neural_nine</code>
</pre>

Next, you must absolutely make sure your Python is compiled for ``ARM64`` when creating ``conda`` or it will not work:

<pre>
<code>(neural_nine) $ conda config --env --set subdir osx-arm64</code>
</pre>

To deactivate the ``conda`` environment:

<pre>
<code>(neural_nine) $ conda deactivate</code>
</pre>

### Requirements

Please install the required ``pip`` packages by running the following commands:

<pre>
<code>(neural_nine) $ pip install --upgrade pip</code>
</pre>

<pre>
<code>(neural_nine) $ pip install -r requirements.txt</code>
</pre>

<i>DISCLAIMER: This is not investing advice. I am not a professional who is qualified in giving any financial advice. This is a Python program purely about coding with financial data from the U.S. stock market.</i>
