package fasttext;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import fasttext.Args.loss_name;
import com.google.common.base.Preconditions;
import org.apache.log4j.Logger;

import java.io.*;

public class Model  implements Serializable {

	private static Logger logger = Logger.getLogger(Model.class);
	static final int NEGATIVE_TABLE_SIZE = 10000000;
	public class Node {
		int parent;
		int left;
		int right;
		long count;
		boolean binary;
	}

	private Args args;
	Dictionary dict;

	private Matrix wi_; // input
	private Matrix wo_; // output
	private Vector hidden_;
//	private Vector output_;
	private int osz_; // output vocabSize
	private java.util.Vector<Integer> negatives;
	private java.util.Vector<java.util.Vector<Integer>> paths;
	private java.util.Vector<java.util.Vector<Boolean>> codes;
	private java.util.Vector<Node> tree;

	public RandomGenerator rng;
	public Model() {
		args = new Args();
		dict = new Dictionary(args);
		wi_ = new Matrix();
		wo_ = new Matrix();
	}
	public void loadModel(String filename) throws IOException {
		DataInputStream dis = null;
		BufferedInputStream bis = null;
		try {
			File file = new File(filename);
			if (!(file.exists() && file.isFile() && file.canRead())) {
				throw new IOException("Model file cannot be opened for loading!");
			}
			bis = new BufferedInputStream(new FileInputStream(file));
			dis = new DataInputStream(bis);

			args.load(dis);
			dict.load(dis);
			wi_.load(dis);
			wo_.load(dis);


			hidden_ = new Vector(args.dim);
//			output_ = new Vector(wo_.m_);
			rng = new Well19937c(1);
			osz_ = wo_.m_;

			logger.info("loadModel done!");
		} finally {
			bis.close();
			dis.close();
		}
	}
	public void predict(String filename) throws IOException {
		java.util.Vector<Integer> line = new java.util.Vector<Integer>();
		java.util.Vector<Integer> labels = new java.util.Vector<Integer>();

		File file = new File(filename);
		if (!(file.exists() && file.isFile() && file.canRead())) {
			throw new IOException("Test file cannot be opened!");
		}
		UniformRealDistribution urd = new UniformRealDistribution(this.rng, 0, 1);
		FileInputStream fis = new FileInputStream(file);
		BufferedReader dis = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
		try {
			String lineString;
			while ((lineString = dis.readLine()) != null) {
				dict.getLine(lineString, line, labels, urd);
				dict.addNgrams(line, args.wordNgrams);
				if (line.size() > 0) {
					int i = this.predict(line);
					System.out.println(lineString + "\t" + dict.getLabel(i));
				} else {
					System.out.println(lineString + "\tn/a");
				}
			}
		} finally {
			dis.close();
			fis.close();
		}
	}
	public String predictStr(String str) throws IOException {
		java.util.Vector<Integer> line = new java.util.Vector<Integer>();
		java.util.Vector<Integer> labels = new java.util.Vector<Integer>();

		UniformRealDistribution urd = new UniformRealDistribution(this.rng, 0, 1);

		String lineString = str;
		dict.getLine(lineString, line, labels, urd);
		dict.addNgrams(line, args.wordNgrams);
		if (line.size() > 0) {
			int i = this.predict(line);
			System.out.println(lineString + "\t" + dict.getLabel(i));
			return dict.getLabel(i);
		} else {
			System.out.println(lineString + "\tn/a");
			return null;
		}
	}

	private int predict(final java.util.Vector<Integer> input) {
		Vector hidden_ = new Vector(args.dim);
		Vector output_ = new Vector(wo_.m_);
		hidden_.zero();
		for (Integer it : input) {
			hidden_.addRow(wi_, it);
		}
		hidden_.mul((float) (1.0 / input.size()));

		if (args.loss == loss_name.hs) {
			float max = -1e10f;
			int argmax = -1;
			dfs(2 * osz_ - 2, 0.0f, max);
			return argmax;
		} else {
			output_.mul(wo_, hidden_);
			return output_.argmax();
		}
	}

	public void dfs(int node, float score, float max) {
		if (score < max)
			return;
		if (tree.get(node).left == -1 && tree.get(node).right == -1) {
			return;
		}
		float f = Utils.sigmoid(wo_.dotRow(hidden_, node - osz_));
		dfs(tree.get(node).left, score + Utils.log(1.0f - f), max);
		dfs(tree.get(node).right, score + Utils.log(f), max);
	}

	public void initTableNegatives(final java.util.Vector<Long> counts) {
		negatives = new java.util.Vector<Integer>(counts.size());
		float z = 0.0f;
		for (int i = 0; i < counts.size(); i++) {
			z += (float) Math.pow(counts.get(i), 0.5f);
		}
		for (int i = 0; i < counts.size(); i++) {
			float c = (float) Math.pow(counts.get(i), 0.5f);
			for (int j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
				negatives.add(i);
			}
		}
		Utils.shuffle(negatives, rng);
	}

	public void setTargetCounts() {
		java.util.Vector<Long> counts = dict.getCounts(Dictionary.entry_type.label);
		Preconditions.checkArgument(counts.size() == osz_);
		if (args.loss == loss_name.ns) {
			initTableNegatives(counts);
		}
		if (args.loss == loss_name.hs) {
			buildTree(counts);
		}
	}

	public void buildTree(final java.util.Vector<Long> counts) {
		paths = new java.util.Vector<java.util.Vector<Integer>>(osz_);
		codes = new java.util.Vector<java.util.Vector<Boolean>>(osz_);
		tree = new java.util.Vector<Node>(2 * osz_ - 1);

		// tree.setSize();
		for (int i = 0; i < 2 * osz_ - 1; i++) {
			Node node = tree.get(i);
			node.parent = -1;
			node.left = -1;
			node.right = -1;
			node.count = 1000000000000000L;// 1e15f;
			node.binary = false;
		}
		for (int i = 0; i < osz_; i++) {
			tree.get(i).count = counts.get(i);
		}
		int leaf = osz_ - 1;
		int node = osz_;
		for (int i = osz_; i < 2 * osz_ - 1; i++) {
			int[] mini = new int[2];
			for (int j = 0; j < 2; j++) {
				if (leaf >= 0 && tree.get(leaf).count < tree.get(node).count) {
					mini[j] = leaf--;
				} else {
					mini[j] = node++;
				}
			}
			tree.get(i).left = mini[0];
			tree.get(i).right = mini[1];
			tree.get(i).count = tree.get(mini[0]).count + tree.get(mini[1]).count;
			tree.get(mini[0]).parent = i;
			tree.get(mini[1]).parent = i;
			tree.get(mini[1]).binary = true;
		}
		for (int i = 0; i < osz_; i++) {
			java.util.Vector<Integer> path = new java.util.Vector<Integer>();
			java.util.Vector<Boolean> code = new java.util.Vector<Boolean>();
			int j = i;
			while (tree.get(j).parent != -1) {
				path.add(tree.get(j).parent - osz_);
				code.add(tree.get(j).binary);
				j = tree.get(j).parent;
			}
			paths.add(path);
			codes.add(code);
		}
	}
}
