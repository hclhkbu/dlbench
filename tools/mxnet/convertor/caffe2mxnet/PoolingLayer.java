package caffe2mxnet;

public final class PoolingLayer extends Layer {

	int kernelDimX;
	int kernelDimY;
	int strideDimX;
	int strideDimY;
	String pool_type;

	public PoolingLayer() {
	}

	public PoolingLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public PoolingLayer(String Name, String Data) {
		super(Name, Data);
	}

	public PoolingLayer(String Name, String Data, int KernelDim, int StrideDim, String Type) {
		this.name = Name;
		this.data = Data;
		this.kernelDimX = KernelDim;
		this.kernelDimY = KernelDim;
		this.strideDimX = StrideDim;
		this.strideDimY = StrideDim;
		this.pool_type = (Type == null || Type.length() < 1) ? "max" : Type;
	}

	public PoolingLayer setKernelDim(int kernelDim) {
		this.kernelDimX = kernelDim;
		this.kernelDimY = kernelDim;
		return this;
	}

	@Deprecated
	public PoolingLayer setKernelDimX(int kernelDimX) {
		this.kernelDimX = kernelDimX;
		return this;
	}

	@Deprecated
	public PoolingLayer setKernelDimY(int kernelDimY) {
		this.kernelDimY = kernelDimY;
		return this;
	}

	public PoolingLayer setStrideDim(int strideDim) {
		this.strideDimX = strideDim;
		this.strideDimY = strideDim;
		return this;
	}

	@Deprecated
	public PoolingLayer setStrideDimX(int strideDimX) {
		this.strideDimX = strideDimX;
		return this;
	}

	@Deprecated
	public PoolingLayer setStrideDimY(int strideDimY) {
		this.strideDimY = strideDimY;
		return this;
	}

	public PoolingLayer setPool_type(String pool_type) {
		if (pool_type.contains("av"))
			this.pool_type = "avg";
		else
			this.pool_type = "max";
		return this;
	}

	public String getKernelDim() {
		return "kernel = (" + Integer.toString(this.kernelDimX) + "," + Integer.toString(kernelDimY) + ")";
	}

	public int getKernelDimX() {
		return kernelDimX;
	}

	public int getKernelDimY() {
		return kernelDimY;
	}

	public String getStrideDim() {
		return "stride = (" + Integer.toString(this.strideDimX) + "," + Integer.toString(strideDimY) + ")";
	}

	public int getStrideDimX() {
		return strideDimX;
	}

	public int getStrideDimY() {
		return strideDimY;
	}

	public String getPoolType() {
		return this.pool_type.length() > 1 ? "pool_type = \"" + this.pool_type + "\"" : "pool_type = \"max\"";
	}

	public String getPool_type() {
		return pool_type;
	}

	@Override
	public String getMXnetString() {
		String str = this.mxPrefix() + "Pooling(" + this.getData();
		str += (this.kernelDimX > 0 && this.kernelDimY > 0) ? ", " + this.getKernelDim() : "";
		str += (this.strideDimX > 0 && this.strideDimY > 0) ? ", " + this.getStrideDim() : "";
		str += ", " + this.getPoolType();
		str += ")";
		return str;
	}

	@Override
	void parseCaffeConfigLines(String[] CaffeConfigLines) {
		for (String line : CaffeConfigLines) {
			if (line.contains("pool:"))
				this.setPool_type(extractContent(line).toLowerCase());
			else if (line.contains("kernel_size:"))
				this.setKernelDim(Integer.parseInt(extractContent(line)));
			else if (line.contains("stride"))
				this.setStrideDim(Integer.parseInt(extractContent(line)));
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
		this.pool_type = "max";
	}

}
