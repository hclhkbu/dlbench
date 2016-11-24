How to add new tools:
	1. Put your tools' scripts directory here and copy the benchmark scrip under common/xxxbm.py (rename it to <your tool name>bm.py) in it.

	2. Make sure xxxbm.py will take all those parameters pre-defined, you may ignore some of them as long as it works.

	3. xxxbm.py should print out the running result which will be taken by benchmark.py and post to the server.

	4. All tests in common/testbm.sh should be passed before make it into use
