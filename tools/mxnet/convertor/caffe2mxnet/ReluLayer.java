package caffe2mxnet;

public final class ReluLayer extends ActivationLayer {

	public ReluLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public ReluLayer() {
	}

	public ReluLayer(String Name, String data) {
		super(Name, data);
	}

	@Override
	void initVariables() {
		this.actType = "relu";
	}

}
