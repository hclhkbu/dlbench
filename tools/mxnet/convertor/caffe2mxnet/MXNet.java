package caffe2mxnet;

import java.util.LinkedList;

public class MXNet {

	private LinkedList<Layer> layers;
	private String netName;

	public MXNet() {
		initLayers();
	}

	public MXNet(String NetName) {
		this.netName = NetName;
		initLayers();
	}

	public MXNet setNetName(String newName) {
		this.netName = newName;
		return this;
	}

	public String getNetName() {
		return this.netName;
	}

	public void initLayers() {
		this.layers = new LinkedList<Layer>();
	}

	public MXNet addLayer(Layer newLayer) {
		this.layers.add(newLayer);
		return this;
	}

	public void printMXNetCode() {
		System.out.println("import mxnet as mx");
		System.out.println();
		for (Layer layer : layers)
			System.out.println(layer.getMXnetString());
	}

	public void printMXNetMissingInfo() {
		System.out.println();
		System.out.println("Missing info:");
		for (Layer layer : layers)
			System.out.println(layer.getMissedLayerInfo());
	}

	public LinkedList<Layer> getLayers() {
		return this.layers;
	}

	public int getNumLayers() {
		return this.layers.size();
	}

	public Layer getTailLayer() {
		return this.layers.getLast();
	}

	public String getTailLayerName() {
		return this.layers.getLast().getNameValue();
	}

	// Test layers
	// public static void main(String[] args) {
	// MXNet net = new MXNet();
	// net.addLayer(new DataLayer("data", "data")).addLayer(new FCLayer("fc1",
	// net.getTailLayerName(), 2048))
	// .addLayer(new PoolingLayer("pool1",
	// net.getTailLayerName()).setKernelDim(3).setStrideDim(1))
	// .addLayer(new ConvolutionLayer("Conv", net.getTailLayerName(), 100, 100,
	// 100, 100))
	// .addLayer(new ReluLayer("relu1", net.getTailLayerName()))
	// .addLayer(new SoftmaxLayer("softmax", net.getTailLayerName()));
	// net.printMXNetCode();
	// }

}
