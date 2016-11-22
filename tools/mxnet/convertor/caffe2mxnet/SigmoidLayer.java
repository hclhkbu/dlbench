package caffe2mxnet;

public final class SigmoidLayer extends ActivationLayer {

	public SigmoidLayer() {
	}

	public SigmoidLayer(String Name, String Data, String[] CaffeConfigLines) {
		super(Name, Data, CaffeConfigLines);
	}

	public SigmoidLayer(String Name, String Data) {
		super(Name, Data);
	}

	@Override
	void initVariables() {
		this.actType = "sigmoid";
	}

}
