package caffe2mxnet;

import java.io.*;

public class Caffe2mxnet {

	private String caffeFilePath;
	private String mxnetFilePath;
	private File caffeConfigFile;

	public Caffe2mxnet(String FilePath) {
		this.caffeFilePath = FilePath;
		caffeConfigFile = new File(this.caffeFilePath);
	}

	public Caffe2mxnet(String caffeFilePath, String mxnetFilePath) {
		this.caffeFilePath = caffeFilePath;
		this.caffeConfigFile = new File(this.caffeFilePath);
		this.mxnetFilePath = mxnetFilePath;
	}

	public void writeOutMXNetCode(MXNet mxnet) {
		Writer writer = null;
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(this.mxnetFilePath), "utf-8"));
			writer.write("import mxnet as mx\n\n\n");
            writer.write("def get_net():\n");
			for (Layer layer : mxnet.getLayers())
				writer.write("\t" + layer.getMXnetString() + "\n");
            writer.write("\treturn " + mxnet.getTailLayerName());
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				writer.close();
			} catch (IOException e) {
			}
		}

	}

	private String extractContent(String line) {
		return line.split(":")[1].replaceAll("\"", "").trim();
	}

	private void addNewLayer(MXNet mxnet, String layerConfig, int lineNum) {
		String[] configLines = layerConfig.split(",");
		String name = "";
		String type = "";
		String top = "";
		String bottom = "";

		int braceCnt = 0;
		for (String line : configLines) {
			if (line.contains("{"))
				braceCnt++;
			if (line.contains("}"))
				braceCnt--;
			if (braceCnt == 1 && line.contains("name"))
				name = extractContent(line);
			else if (braceCnt == 1 && line.contains("type"))
				type = extractContent(line);
			else if (braceCnt == 1 && line.contains("bottom"))
				bottom = extractContent(line);
			else if (braceCnt == 1 && line.contains("top"))
				top = extractContent(line);
			if (name.length() > 1 && type.length() > 1 && bottom.length() > 1 && top.length() > 1)
				break;
		}

		// bottom can be configured as previous layer
		if (mxnet.getNumLayers() > 0 && mxnet.getTailLayer() instanceof ActivationLayer)
			bottom = mxnet.getTailLayerName();

		switch (type) {
		case "Convolution":
			mxnet.addLayer(new ConvolutionLayer(name, bottom, configLines));
			break;
		case "Data":
			mxnet.addLayer(new DataLayer(name, bottom, configLines));
			break;
		case "InnerProduct":
			mxnet.addLayer(new FCLayer(name, bottom, configLines));
			break;
		case "Pooling":
			mxnet.addLayer(new PoolingLayer(name, bottom, configLines));
			break;
		case "BatchNorm":
			mxnet.addLayer(new BatchNormLayer(name, bottom, configLines));
			break;
		case "ReLU":
			mxnet.addLayer(new ReluLayer(name, bottom, configLines));
			break;
		case "Sigmoid":
			mxnet.addLayer(new SigmoidLayer(name, bottom, configLines));
			break;
		case "SoftmaxWithLoss":
			mxnet.addLayer(new SoftmaxLayer(name, bottom, configLines));
			break;
		case "Softmax":
			mxnet.addLayer(new SoftmaxLayer(name, bottom, configLines));
			break;
		default:
			System.err.println("[Near line " + Integer.toString(lineNum) + "] Unknown layer type: " + type + " name: "
					+ name + " input data from: " + bottom);
			break;
		}

	}

	protected MXNet caffeNet2mxnet(File caffeNetFile) throws IOException {
		MXNet mxnet = new MXNet();
		FileInputStream fis = new FileInputStream(caffeNetFile);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		String layerConfig = "";
		int braceCounter = 0;
		int lineCounter = 0;

		String line = null;
		while ((line = br.readLine()) != null) {
			lineCounter++;
			if (line.length() < 1)
				continue;
			if (lineCounter < 2 && line.contains("name:")) {
				System.out.println("Net name: " + extractContent(line));
				mxnet.setNetName(extractContent(line));
				continue;
			}

			if (line.contains("{"))
				braceCounter++;
			else if (line.contains("}"))
				braceCounter--;
			layerConfig += line.trim() + ",";
			if (braceCounter < 0) {
				System.err.println("Error reading Caffe config file near line " + Integer.toString(lineCounter) + "!!");
				System.exit(-1);
			} else if (braceCounter == 0 && layerConfig.length() > 4) {
				this.addNewLayer(mxnet, layerConfig, lineCounter);
				layerConfig = "";
			}

		}

		br.close();

		return mxnet;
	}

	public MXNet caffe2mxnet() {
		try {
			return caffeNet2mxnet(this.caffeConfigFile);
		} catch (IOException e) {
			System.err.println("Failed to open caffe config file!");
			e.printStackTrace();
		}
		return null;
	}

	private static void printHelp() {
		System.out.println("Caffe2mxnet, naive caffe network config convertor.\n");
		System.out.println("args: ");
		System.out.println("\t -caffe <path to caffe config input file> : required");
		// System.out.println("\t -mxnet <path to mxnet config output file> :
		// optional");
		System.out.println("The output path will be the same as input file path.");
		System.exit(0);
	}

	public static void main(String[] args) {
		String inPath = "";
		String outPath = "";

		for (int i = 0; i < args.length; i++) {
			if (args[i].contains("-caffe")) {
				try {
					inPath = args[i + 1];
				} catch (Exception e) {
					printHelp();
				}
			}
		}
		if (inPath.length() < 1)
			printHelp();
		// else if (outPath.length() < 1)
		outPath = inPath.replace("prototxt", "py");
		Caffe2mxnet c2m = new Caffe2mxnet(inPath, outPath);
		MXNet mxnet = c2m.caffe2mxnet();
		c2m.writeOutMXNetCode(mxnet);
		mxnet.printMXNetMissingInfo();
		// Debug:
		// // Caffe2mxnet c2m = new
		// //
		// Caffe2mxnet("F:\\gitDir\\hkbu-benchmark\\caffe\\fc\\ffn26752-b64.prototxt");
		// // Caffe2mxnet c2m = new
		// //
		// Caffe2mxnet("F:\\gitDir\\hkbu-benchmark\\caffe\\cnn\\alexnet\\alexnet-b64.prototxt");
		// Caffe2mxnet c2m = new
		// Caffe2mxnet("F:\\gitDir\\hkbu-benchmark\\caffe\\cnn\\resnet\\resnet-b64.prototxt",
		// "F:\\gitDir\\hkbu-benchmark\\caffe\\cnn\\resnet\\resnet-b64.py");
		// // Caffe2mxnet c2m = new
		// //
		// Caffe2mxnet("/home/comp/pengfeixu/hkbu-benchmark/caffe/cnn/resnet/resnet-b64.prototxt");
		// MXNet mxnet = c2m.caffe2mxnet();
		// mxnet.printMXNetMissingInfo();
		// mxnet.writeOutMXNetCode(c2m.mxnetFilePath);
	}

}
