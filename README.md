# Grasp-and-Lift-EEG-Detection
Identify Hand Motion from EEG recordings

Data can be found at the projects's Kaggle.com page:
https://www.kaggle.com/c/grasp-and-lift-eeg-detection

README for Grasp and Lift EEG Detection

Sagar Patel

System Utilized:
- For normal (Non-hadoop) program run:
	- Microsoft Azure's A11 VM instance (16 Core, 112GB RAM)
    	- OS:          Ubuntu 14.04
	- Packages:	numpy,
			scipy,
			sklearn,
			multiprocessing (for parallelization)
			pandas  (pip install pandas OR sudo apt-get install python pandas),
			neon 
			Install NEON from source------> git clone https://github.com/NervanaSystems/neon.git
							cd neon
							make install
- For Hadoopable program:
	- Microsoft's HDInsight framework (1 master node, 2 data nodes)
	- OS: 	     Each node has Ubuntu 12.04
	- Packages:        Same as above


Genaral Approach:
- The general program takes data using NEON's batch input method which results in lesser memory requirements. (The RAM utilisation was   about 11GB, which would have been 4x that, had I not been using batch processing)
- Moving on, chunks of data get Bandpass filtered afterwards. 
- Initially CSP (Common Spatial Filters) were being used with Logistics regression, which were yielding low results.
- So, afterwards it was decided to use Convolutional Neural nets for training and leaving it upto the convnets to figure out       	  underlying function.
- Pool function from multiprocessing package has been used to parallelize the training as well as testing over data for each 12 	  subjects.
- All the reading of data and batch signal processing has been performed in the class: GalData
- Convolutional Network: 9 layers
	- Data layer
	- Dropout Layer (0.95)
	- Convolutional Layer (rectilinear activation function)
	- Dropout layer (0.8)
	- Pooling Layer 
	- Dropout Layer (0.6)
	- FC layer 1 (rectilinear activation)
	- FC layer 2 (Logistic activation)
- The code runs in two different modes:
	1) Validation: If validate = True (uses only one window of 1024 samples)
	2) Prediction: If validate = False (default case, uses 4 windows [768, 1024, 1280, 1536])
- Sampling: Uniform sampling frequency = 4

NOTE: For faster execution increase the sampling frequency to 32 and only test code for one window

Hadoop (HDInsight) Approach:
- Multiple Hadoop nodes are considered to be very helpful in training deeper and wider convolutional nets in parallel
- Unfortunately I couldn't find a proper documentation on how to implement that.
- So I ended up utilizing only a single cluster out of the four clusters that I had
- HDInsight uses Hadoop Streaming for Python just like Cloudera, however there is one caveat:
	- Inherent working of both framework is quite different.
	- In case of Cloudera, one can do: " hs mapper.py reducer.py input_data output_file "
		  This processes the mapper data and it should be printed (i.e. std.out method)
		  The reducer uses std.in method to collect the data streaming in from mapper.
	- HDInsight also works in almost same way, except that we can pass whole trained network and processed data without shuffle
		  and sort (which is not required in this case). Hence, here I am passing the 'self' object which contains the trained n/w.
NOTE: Due to the previous point the Hadoopable version only works with HDInsight and not with Cloudera Hadoop.
      So, if you do not have HDInsight subscription, then please run the full code.

DATA: Data for this project can be downloaded from Kaggle.com
      URL of competition: https://www.kaggle.com/c/grasp-and-lift-eeg-detection
      However, please note that you must be a kaggle member to be able to download the data

Running code and Directory structure:

	The train and test directory should be kept inside directory named 'input'
	The code should be kept inside directory named 'src'

Problems faced:
	- Convnets can be easily trained on multicore processors, and hence GPUs with thousands of cores are optimal for such    		  applications. 
	- Unfortunately Azure doesn't have GPU support so I had to parallelize my code in 16 cores available to me.
	- So, it took 12.5 hours to run the final version of the code.
	- The original code integrated the training and testing of convnets very well. Hence, it was really hard to make the code 		  hadoopable.	
	- The HDInsight pricing is way too high. Which led to two of 150$/month subscriptions utilization. 
	- Unavailability of further processing power prevented further tweaking/testing.
	
Results: AUC = 0.94039
	 Position on Leaderboard = 49 (At the time of writing) 

Special Thanks:
	Alexandre Barachant (For providing link for CSP filters, irrespective of the fact that I didn't use it in final code)
	Tim Hotchberg (For suggesting me the NEON library for convnets an the batch processing of data)

Overall, this was a pretty fun project in which I get to test the signal processing and ML theoretical knowledge that I acquired over time.
Also this was my first big Python project, so got to learn some caveats of the language. The immense amount of time gone into doing this seems well-invedted with my very first Kaggle project getting in Top 50 out of about 390 total Participants.
