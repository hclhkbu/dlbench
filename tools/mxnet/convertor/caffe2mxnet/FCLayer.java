package caffe2mxnet;

public final class FCLayer extends Layer {

	int num_hidden;

	public FCLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public FCLayer() {
		super();
	}

	public FCLayer(String Name, String Data) {
		super(Name, Data);
	}

	public FCLayer(String name, String data, int num_hidden) {
		this.name = name;
		this.data = data;
		this.num_hidden = num_hidden;
	}

	public String getNumHidden() {
		return "num_hidden = " + Integer.toString(num_hidden);
	}

	public int getNum_hidden() {
		return num_hidden;
	}

	public FCLayer setNum_hidden(int num_hidden) {
		this.num_hidden = num_hidden;
		return this;
	}

	@Override
	public String getMXnetString() {
		if (this.num_hidden < 1) {
			System.err.println(
					"[Layer " + this.getNameValue() + "] Number of hidden layers must be set! Program terminated!");
			System.exit(-1);
		}
		return this.mxPrefix() + "FullyConnected(" + this.getData() + ", " + this.getNumHidden() + ")";
	}

	@Override
	void parseCaffeConfigLines(String[] CaffeConfigLines) {
		for (String line : CaffeConfigLines) {
			if (line.contains("num_output"))
				this.setNum_hidden(Integer.parseInt(extractContent(line)));
			else 
				addToMissionInfo(line);
		}
	}

	@Override
	void initVariables() {
		this.num_hidden = -1;
	}

}
