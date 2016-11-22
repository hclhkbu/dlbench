package caffe2mxnet;

public final class BatchNormLayer extends Layer {
	// As defined here:
	// https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L500
	final static float DEFAULT_EPS = 2e-5f; // Should be 1e-5f but cudnn doesn't
											// allow
	final static float DEFAULT_MOMENTUM = 0.9f;
	final static boolean DEFAULT_FIX_GAMMA = true;
	final static boolean DEFAULT_USE_GLOBAL_STATS = false;

	private float eps, momentum;
	private boolean fixGamma, useGlobalStats;

	public BatchNormLayer() {
	}

	public BatchNormLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	void initVariables() {
		// Default value set as
		this.eps = DEFAULT_EPS;
		this.momentum = DEFAULT_MOMENTUM;
		this.fixGamma = DEFAULT_FIX_GAMMA;
		this.useGlobalStats = DEFAULT_USE_GLOBAL_STATS;
	}

	public String getEps() {
		return "eps = " + Float.toString(this.eps);
	}

	public String getMomentum() {
		return "momentum = " + Float.toString(this.momentum);
	}

	public String getFixGamma() {
		return this.fixGamma ? "fix_gamma = True" : "fix_gamma = False";
	}

	public String getUseGlobalStats() {
		return useGlobalStats ? "use_global_stats = True" : "use_global_stats = Flase";
	}

	public float getEpsValue() {
		return eps;
	}

	public float getMomentumValue() {
		return momentum;
	}

	public boolean isFixGammaValue() {
		return fixGamma;
	}

	public boolean isUseGlobalStatsValue() {
		return useGlobalStats;
	}

	public BatchNormLayer(String Name, String Data) {
		super(Name, Data);
	}

	@Override
	void parseCaffeConfigLines(String[] CaffeConfigLines) {
		for (String line : CaffeConfigLines) {
			if (line.contains("use_global_stats"))
				this.useGlobalStats = Boolean.parseBoolean(extractContent(line));
			else if (line.contains("eps"))
				this.eps = Float.parseFloat(extractContent(line));
			else if (line.contains("moving_average"))
				this.momentum = Float.parseFloat(extractContent(line));
			else
				addToMissionInfo(line);
		}
	}

	@Override
	public String getMXnetString() {
		String str = this.mxPrefix() + "BatchNorm(" + this.getData();
		str += ", " + this.getEps() + ", " + this.getMomentum() + ", " + this.getUseGlobalStats() + ", "
				+ this.getFixGamma();
		str += ")";
		return str;
	}

}
