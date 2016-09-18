package fasttext;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class Args implements Serializable{

	public enum model_name {
		cbow(1), sg(2), sup(3);

		private int value;

		private model_name(int value) {
			this.value = value;
		}

		public int getValue() {
			return this.value;
		}

		public static model_name fromValue(int value) throws IllegalArgumentException {
			try {
				value -= 1;
				return model_name.values()[value];
			} catch (ArrayIndexOutOfBoundsException e) {
				throw new IllegalArgumentException("Unknown model_name enum value :" + value);
			}
		}
	}

	public enum loss_name {
		hs(1), ns(2), softmax(3);
		private int value;

		private loss_name(int value) {
			this.value = value;
		}

		public int getValue() {
			return this.value;
		}

		public static loss_name fromValue(int value) throws IllegalArgumentException {
			try {
				value -= 1;
				return loss_name.values()[value];
			} catch (ArrayIndexOutOfBoundsException e) {
				throw new IllegalArgumentException("Unknown loss_name enum value :" + value);
			}
		}
	}

	public int lrUpdateRate = 100;
	public int dim = 100;
	public int ws = 5;
	public int epoch = 5;
	public int minCount = 5;
	public int neg = 5;
	public int wordNgrams = 1;
	public loss_name loss = loss_name.ns;
	public model_name model = model_name.sg;
	public int bucket = 2000000;
	public int minn = 3;
	public int maxn = 6;
	public double t = 1e-4;
	public String label = "__label__";

	public void load(InputStream input) throws IOException {
		dim = IOUtil.readInt(input);
		ws = IOUtil.readInt(input);
		epoch = IOUtil.readInt(input);
		minCount = IOUtil.readInt(input);
		neg = IOUtil.readInt(input);
		wordNgrams = IOUtil.readInt(input);
		loss = loss_name.fromValue(IOUtil.readInt(input));
		model = model_name.fromValue(IOUtil.readInt(input));
		bucket = IOUtil.readInt(input);
		minn = IOUtil.readInt(input);
		maxn = IOUtil.readInt(input);
		lrUpdateRate = IOUtil.readInt(input);
		t = IOUtil.readDouble(input);
	}
}
