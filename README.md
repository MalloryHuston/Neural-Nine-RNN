# NeuralNine RNN
This algorithm predicts stock prices in Python using recurrent neural networks and machine learning. Practice run from YouTuber <a href="https://www.youtube.com/watch?v=PuZY9q-aKLw">NeuralNine</a>. <b>[Published on February 2nd, 2021]</b>

### VIRTUAL ENVIRONMENT

It is highly recommended to run this program in a ``conda`` environment. Download and install it by following these steps:

1. Download the ``.sh`` installer by opening up the terminal and running the following command:
<pre>
<code>$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh</code>
</pre>

2. Install it by running the following:
<pre>
<code>$ bash ~/Miniconda3-latest-MacOSX-arm64.sh</code>
</pre>

Now that has already been set up, you can create a ``conda`` environment and set up its environment as ``Python 3.9.1``:
<pre>
<code>$ conda create -n neural_nine python=3.9.1</code>
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

Please install the required ``pip`` packages by running the following command:

<pre>
<code>(neural_nine) $ pip install -r requirements.txt</code>
</pre>

<b><i>DISCLAIMER: This is not investing advice. I am not a professional who is qualified in giving any financial advice. This is a Python program purely about coding with financial data from the U.S. stock market.</i></b>
