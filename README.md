# NeuralNine RNN
This algorithm predicts stock prices in Python using recurrent neural networks and machine learning. Practice run from YouTuber <a href="https://www.youtube.com/watch?v=PuZY9q-aKLw">NeuralNine</a>. <b>[Published on February 2nd, 2021]</b>

### VIRTUAL ENVIRONMENT

It is highly recommended to run this program in a ``conda`` environment. Download and install it by following these steps:

1. Download the ``.sh`` installer by opening up the terminal and running the following commands:
<pre>
<code>mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh</code>
</pre>

2. After installing, close and reopen your terminal application or refresh it by running the following command:
<pre>
<code>source ~/miniconda3/bin/activate</code>
</pre>

3. Then, initialize conda on all available shells by running the following command:
<pre>
<code>conda init --all</code>
</pre>

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

### REQUIREMENTS

Please install the required ``pip`` packages by running the following commands:

<pre>
<code>(neural_nine) $ pip install --upgrade pip</code>
</pre>

<pre>
<code>(neural_nine) $ pip install -r requirements.txt</code>
</pre>

<i>DISCLAIMER: This is not investing advice. I am not a professional who is qualified in giving any financial advice. This is a Python program purely about coding with financial data from the U.S. stock market.</i>
