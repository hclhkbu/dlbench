package caffe2mxnet;

public final class SoftmaxLayer extends Layer {

	public SoftmaxLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public SoftmaxLayer() {

	}

	public SoftmaxLayer(String Name, String Data) {
		super(Name, Data);
	}

	@Override
	public String getMXnetString() {
		return this.mxPrefix() + "SoftmaxOutput(" + this.getData() + ", name = \'softmax\')";
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
