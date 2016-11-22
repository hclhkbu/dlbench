package caffe2mxnet;

public abstract class Layer {

	static final String MX_SYM_PREFIX = " = mx.symbol.";
	String name; // variable name of the layer, not the name of the layer
	String data; // input source
	String missedLayerInfo = "";

	public Layer() {
		// Default constructor
	}

	public Layer(String Name, String Data, String[] CaffeConfigLines) {
		this.name = Name;
		this.data = Data;
		this.missedLayerInfo += this.name + " ";
		initVariables();
		this.parseCaffeConfigLines(CaffeConfigLines);
	}

	abstract void initVariables();

	abstract void parseCaffeConfigLines(String[] CaffeConfigLines);

	public Layer(String Name, String Data) {
		this.name = Name;
		this.data = Data;
	}

	public Layer setName(String Name) {
		this.name = Name;
		return this;
	}

	public String getData() {
		return "data = " + this.data;
	}

	public String getDataValue() {
		return data;
	}

	public Layer setData(String data) {
		this.data = data;
		return this;
	}

	public String getNameValue() {
		return name;
	}

	String mxPrefix() {
		return this.name + MX_SYM_PREFIX;
	}

	String extractContent(String line) {
		return line.split(":")[1].replaceAll("\"", "").trim();
	}

	String getMissedLayerInfo() {
		return this.missedLayerInfo;
	}

	int tapCnt = 0;

	void addToMissionInfo(String line) {
		if (line.contains("}"))
			this.tapCnt--;
		for (int i = 0; i < tapCnt; i++)
			this.missedLayerInfo += "\t";
		if (line.contains("{"))
			this.tapCnt++;
		this.missedLayerInfo += line + "\n";
	}

	public abstract String getMXnetString();
}
