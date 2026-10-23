# NeuralNine RNN

***DISCLAIMER - PLEASE READ BEFORE DOWNLOADING:***

*This is **<ins>NOT</ins>** investing advice. I am **<ins>NOT</ins>** a professional who is qualified in giving any financial advice. This is a Python program purely about coding financial data from the New York Stock Exchange.*

This algorithm attempts to predict future stock prices by using recurrent neural networks and machine learning. Tremendous shout out to [NeuralNine](https://www.youtube.com/watch?v=PuZY9q-aKLw) who taught me how to get started on a project like this.

## Quickstart Install Instructions

It is highly recommended that you run this program in a custom `conda` environment. You can find further instructions on how to install `miniconda.sh` on your computer based on its operating system [here.](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

Now, download and install `miniconda.sh` by following these steps:

### 1. Run the following commands to download and install the latest macOS installer for Apple Silicon. Line by line, write out these four commands:

```bash
# Create a new folder named `miniconda3` in your home directory.
mkdir -p ~/miniconda3

# Download the macOS Miniconda installation script for your Apple Silicon architecture and save the script as `miniconda.sh` in the `miniconda3` folder.
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh

# Run the `miniconda.sh` installation script in silent mode using `Bash`.
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 

# Remove the `miniconda.sh` installation script file after the installation is complete.
rm ~/miniconda3/miniconda.sh
```

### 2. After installing, close and reopen your terminal application or refresh it by running the following command:

```bash
source ~/miniconda3/bin/activate
```

### 3. Next, you'll need to initialize `conda` on all available shells by running the following command:

```bash
conda init --all
```

## Virtual Environment

Now that it has already been set up, create your own `conda` environment and set up its interpreter as `Python 3.10`:

```bash
conda create -n neural_nine python=3.10
```

Now you have set up an isolated environment called `neural_nine`, a sandbox-like structure to install everything mentioned in the `requirements.txt` file. But first, please activate your `conda` environment by entering the following command:

```bash
conda activate neural_nine
```

Next, simply confirm that your Python in this `conda` environment is the correct version `Python 3.10.X`:

```bash
python --version
```

Before running the code, ensure your `conda` environment uses the `ARM64` version of Python. Using any other version will cause the code to break. You can easily do so by running this command here:

```bash
conda config --env --set subdir osx-arm64
```

## Requirements

Please install the required `pip` packages by running the following commands:

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install -r requirements.txt
```

Once you're all done for today, please exit the `conda` environment by simply running:

```bash
conda deactivate
```
