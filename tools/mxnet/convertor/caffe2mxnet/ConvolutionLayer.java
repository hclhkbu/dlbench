package caffe2mxnet;

public final class ConvolutionLayer extends Layer {
	int kernelDimX;
	int kernelDimY;
	int strideDimX;
	int strideDimY;
	int numFilter;
	int padDimX;
	int padDimY;

	public ConvolutionLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public ConvolutionLayer() {
		super();
	}

	public ConvolutionLayer(String Name, String Data) {
		super(Name, Data);
	}

	public ConvolutionLayer(String Name, String Data, int NumFilter, int KernelDim, int StrideDim, int PadDim) {
		this.name = Name;
		this.data = Data;
		this.numFilter = NumFilter;
		this.kernelDimX = KernelDim;
		this.kernelDimY = KernelDim;
		this.strideDimX = StrideDim;
		this.strideDimY = StrideDim;
		this.padDimX = PadDim;
		this.padDimY = PadDim;
	}

	public String getKernelDim() {
		return "kernel = (" + Integer.toString(this.kernelDimX) + ", " + Integer.toString(this.kernelDimY) + ")";
	}

	public String getStrideDim() {
		return "stride = (" + Integer.toString(this.strideDimX) + ", " + Integer.toString(this.strideDimY) + ")";
	}

	public String getPadDim() {
		return "pad = (" + Integer.toString(this.padDimX) + ", " + Integer.toString(this.padDimY) + ")";
	}

	public String getNumFilter() {
		return "num_filter = " + Integer.toString(this.numFilter);
	}

	public ConvolutionLayer setKernelDim(int kernelDim) {
		this.kernelDimX = kernelDim;
		this.kernelDimY = kernelDim;
		return this;
	}

	@Deprecated // Squared kernel are used usually, use setKernelDim
	public ConvolutionLayer setKernelDimX(int kernelDimX) {
		this.kernelDimX = kernelDimX;
		return this;
	}

	@Deprecated // Squared kernel are used usually
	public ConvolutionLayer setKernelDimY(int kernelDimY) {
		this.kernelDimY = kernelDimY;
		return this;
	}

	public ConvolutionLayer setStrideDim(int strideDim) {
		this.strideDimX = strideDim;
		this.strideDimY = strideDim;
		return this;
	}

	@Deprecated
	public ConvolutionLayer setStrideDimX(int strideDimX) {
		this.strideDimX = strideDimX;
		return this;
	}

	@Deprecated
	public ConvolutionLayer setStrideDimY(int strideDimY) {
		this.strideDimY = strideDimY;
		return this;
	}

	public ConvolutionLayer setNumFilter(int numFilter) {
		this.numFilter = numFilter;
		return this;
	}

	public ConvolutionLayer setPadDim(int padDim) {
		this.padDimX = padDim;
		this.padDimY = padDim;
		return this;
	}

	@Deprecated
	public ConvolutionLayer setPadDimX(int padDimX) {
		this.padDimX = padDimX;
		return this;
	}

	@Deprecated
	public ConvolutionLayer setPadDimY(int padDimY) {
		this.padDimY = padDimY;
		return this;
	}

	@Override
	public String getMXnetString() {
		String str = mxPrefix() + "Convolution(" + "data = " + this.data;
		str += (this.kernelDimX > 0 && this.kernelDimY > 0) ? ", " + this.getKernelDim() : "";
		str += (this.strideDimX > 0 && this.strideDimY > 0) ? ", " + this.getStrideDim() : "";
		str += (this.padDimX > 0 && this.padDimY > 0) ? "," + this.getPadDim() : "";
		str += (this.numFilter > 0) ? ", " + this.getNumFilter() : "";
		str = str + ")";
		return str;
	}

	@Override
	void parseCaffeConfigLines(String[] CaffeConfigLines) {
		for (String line : CaffeConfigLines) {
			if (line.contains("kernel_size"))
				this.setKernelDim(Integer.parseInt(extractContent(line)));
			else if (line.contains("stride"))
				this.setStrideDim(Integer.parseInt(extractContent(line)));
			else if (line.contains("pad"))
				this.setPadDim(Integer.parseInt(extractContent(line)));
			else if (line.contains("num_output"))
				this.setNumFilter(Integer.parseInt(extractContent(line)));
			else 
				addToMissionInfo(line);
		}
	}

	@Override
	void initVariables() {
		this.kernelDimX = -1;
		this.kernelDimY = -1;
		this.strideDimX = -1;
		this.strideDimY = -1;
		this.numFilter = -1;
		this.padDimX = -1;
		this.padDimY = -1;
	}
}
