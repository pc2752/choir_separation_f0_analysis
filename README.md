<h1>Multi Pitch Extraction and Source Separation Based For SATB Choirs</h1>

<h2>Pritish Chandna, Helena Cuesta, Emilia Gómez</h2>

<h2>Music Technology Group, Universitat Pompeu Fabra, Barcelona</h2>

This repository contains the source code for multi-pitch extration, based on DeepSalience[1] and for source separation for the case of SATB choirs.

<h3>Installation</h3>
To install, clone the repository and use <pre><code>pip install requirements.txt </code></pre> to install the packages required.

 The main code is in the *main.py* file.  
 

<h3>Training and inference</h3>


Once setup, you can run the followinf commands. 
To train the model: 
<pre><code>python main.py -t</code></pre>. 
To display evaluation metrics for a particular file:
Use <pre><code>python main.py -e <i>filename</i></code></pre> 
To save evaluation metrics for the entrie dataset and save to a csv file:
Use <pre><code>python main.py -v <i>filename</i></code></pre> 
to evaluate a wavefile and save multi pitch output to a csv file:
Use <pre><code>python main.py -w <i>filename</i> <i>outfile</i></code></pre> 
to evaluate all wavefiles in a folder and save multi pitch output to a csv file:
Use <pre><code>python main.py -w <i>foldername</i> <i>outfolder</i></code></pre> 
<h3>Evaluation</h3> 


 We will further update the repository in the coming months. 


<h2>Acknowledgments</h2>
The TITANX used for this research was donated by the NVIDIA Corporation. This work is partially supported by the Towards Richer Online Music Public-domain Archives <a href="https://trompamusic.eu/" rel="nofollow">(TROMPA)</a> (H2020 770376) European project.

[1] Bittner, Rachel M., et al. "Deep Salience Representations for F0 Estimation in Polyphonic Music." ISMIR. 2017.
