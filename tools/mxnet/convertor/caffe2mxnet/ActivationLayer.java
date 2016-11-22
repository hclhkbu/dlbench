package caffe2mxnet;

public abstract class ActivationLayer extends Layer {

	String actType;

	public ActivationLayer() {
	}

	public ActivationLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public ActivationLayer(String Name, String Data) {
		super(Name, Data);
	}

	@Override
	void parseCaffeConfigLines(String[] CaffeConfigLines) {
		for (String line : CaffeConfigLines)
			addToMissionInfo(line);
	}

	@Override
	public String getMXnetString() {
		return this.mxPrefix() + "Activation(" + this.getData() + ", act_type=\"" + this.actType + "\")";
	}

}
