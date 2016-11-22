package caffe2mxnet;

public final class DataLayer extends Layer {

	public DataLayer() {
		super();
		// TODO Auto-generated constructor stub
	}

	public DataLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public DataLayer(String Name, String Data) {
		super(Name, Data);
	}

	@Override
	public String getMXnetString() {
		return this.mxPrefix() + "Variable(name = \'data\')";
	}

	@Override
	void parseCaffeConfigLines(String[] CaffeConfigLines) {
		for(String line : CaffeConfigLines)
			addToMissionInfo(line);
	}

	@Override
	void initVariables() {
		return;
	}

}
