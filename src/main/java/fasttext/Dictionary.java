package fasttext;

import java.io.*;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Vector;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.log4j.Logger;

import fasttext.Args.model_name;
import com.google.common.base.Preconditions;
import com.google.common.primitives.UnsignedInteger;

import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;

public class Dictionary implements Serializable{

	private static Logger logger = Logger.getLogger(Dictionary.class);

	private static final int MAX_VOCAB_SIZE = 10000000;
	private static final int MAX_LINE_SIZE = 1024;

	// private static final String EOS = "</s>";
	private static final String BOW = "<";
	private static final String EOW = ">";

	public enum entry_type {
		word(0), label(1);

		private int value;

		private entry_type(int value) {
			this.value = value;
		}

		public int getValue() {
			return this.value;
		}

		public static entry_type fromValue(int value) throws IllegalArgumentException {
			try {
				return entry_type.values()[value];
			} catch (ArrayIndexOutOfBoundsException e) {
				throw new IllegalArgumentException("Unknown entry_type enum value :" + value);
			}
		}

		@Override
		public String toString() {
			return value == 0 ? "word" : value == 1 ? "label" : "unknown";
		}
	}

	public class entry implements Serializable{
		String word;
		long count;
		entry_type type;
		Vector<Integer> subwords;
	}

	private Vector<entry> words_;
	private Vector<Float> pdiscard_;
	private Long2IntOpenHashMap word2int_; // Map<Long, Integer>
	private int size_;
	private int nwords_;
	private int nlabels_;
	private long ntokens_;

	private Args args;

	public Dictionary(Args args) {
		size_ = 0;
		nwords_ = 0;
		nlabels_ = 0;
		ntokens_ = 0;
		word2int_ = new Long2IntOpenHashMap(MAX_VOCAB_SIZE);
		((Long2IntOpenHashMap) word2int_).defaultReturnValue(-1);
		words_ = new Vector<entry>(MAX_VOCAB_SIZE);
		this.args = args;
	}

	public long find(final String w) {
		long h = hash(w) % MAX_VOCAB_SIZE;
		entry e = null;
		while (word2int_.get(h) != -1 && ((e = words_.get(word2int_.get(h))) != null && !w.equals(e.word))) {
			h = (h + 1) % MAX_VOCAB_SIZE;
		}
		return h;
	}

	/**
	 * String FNV-1a Hash
	 *
	 * @param str
	 * @return
	 */
	public static long hash(final String str) {
		int h = (int) 2166136261L;// 0xffffffc5;
		for (byte strByte : str.getBytes()) {
			h = (h ^ strByte) * 16777619; // FNV-1a
			// h = (h * 16777619) ^ strByte; //FNV-1
		}
		return UnsignedInteger.fromIntBits(h).longValue();
	}

	public final Vector<Integer> getNgrams(int i) {
		Preconditions.checkArgument(i >= 0);
		Preconditions.checkArgument(i < nwords_);
		return words_.get(i).subwords;
	}

	public final Vector<Integer> getNgrams(final String word) {
		Vector<Integer> ngrams = new Vector<Integer>();
		int i = getId(word);
		if (i >= 0) {
			ngrams = words_.get(i).subwords;
		} else {
			computeNgrams(BOW + word + EOW, ngrams);
		}
		return ngrams;
	}

	public int getId(final String w) {
		long h = find(w);
		return word2int_.get(h);
	}

	public entry_type getType(int id) {
		Preconditions.checkArgument(id >= 0);
		Preconditions.checkArgument(id < size_);
		return words_.get(id).type;
	}

	public boolean discard(int id, float rand) {
		Preconditions.checkArgument(id >= 0);
		Preconditions.checkArgument(id < nwords_);
		if (args.model == model_name.sup)
			return false;
		return rand > pdiscard_.get(id);
	}

	private boolean charMatches(char ch) {
		if (ch == ' ') {
			return true;
		} else if (ch == '\t') {
			return true;
		} else if (ch == '\n') {
			return true;
		} else if (ch == '\f') { // \x0B
			return true;
		} else if (ch == '\r') {
			return true;
		}
		return false;
	}

	public void computeNgrams(final String word, Vector<Integer> ngrams) {
		for (int i = 0; i < word.length(); i++) {
			StringBuilder ngram = new StringBuilder();
			if (charMatches(word.charAt(i))) {
				continue;
			}
			for (int j = i, n = 1; j < word.length() && n <= args.maxn; n++) {
				ngram.append(word.charAt(j++));
				while (j < word.length() && charMatches(word.charAt(i))) {
					ngram.append(word.charAt(j++));
				}
				if (n >= args.minn) {
					int h = (int) (hash(ngram.toString()) % args.bucket);
					if (h < 0) {
						System.err.println("computeNgrams h<0: " + h + " on word: " + word);
					}
					ngrams.add(nwords_ + h);
				}
			}
		}
	}

	public void initNgrams() {
		for (int i = 0; i < size_; i++) {
			String word = BOW + words_.get(i).word + EOW;
			entry e = words_.get(i);
			if (e.subwords == null) {
				e.subwords = new Vector<Integer>();
			}
			e.subwords.add(i);
			computeNgrams(word, e.subwords);
		}
	}

	public String getLabel(int lid) {
		Preconditions.checkArgument(lid >= 0);
		Preconditions.checkArgument(lid < nlabels_);
		return words_.get(lid + nwords_).word;
	}

	public void initTableDiscard() {
		pdiscard_ = new Vector<Float>(size_);
		for (int i = 0; i < size_; i++) {
			float f = (float) (words_.get(i).count) / (float) ntokens_;
			pdiscard_.add((float) (Math.sqrt(args.t / f) + args.t / f));
		}
	}

	public Vector<Long> getCounts(entry_type type) {
		Vector<Long> counts = new Vector<Long>(words_.size());
		for (entry w : words_) {
			if (w.type == type)
				counts.add(w.count);
		}
		return counts;
	}

	private transient Comparator<entry> entry_comparator = new Comparator<entry>() {
		@Override
		public int compare(entry o1, entry o2) {
			int cmp = o1.type.value > o2.type.value ? +1 : o1.type.value < o2.type.value ? -1 : 0;
			if (cmp == 0) {
				cmp = o1.count > o2.count ? +1 : o1.count < o2.count ? -1 : 0;
			}
			return cmp;
		}
	};

	public void addNgrams(Vector<Integer> line, int n) {
		int line_size = line.size();
		for (int i = 0; i < line_size; i++) {
			int h = line.get(i);
			for (int j = i + 1; j < line_size && j < i + n; j++) {
				h = h * 116049371 + line.get(j);
				line.add(nwords_ + (h % args.bucket));
			}
		}
	}

	public int getLine(String line, Vector<Integer> words, Vector<Integer> labels, UniformRealDistribution urd)
			throws IOException {
		int ntokens = 0;
		words.clear();
		labels.clear();
		if (line != null) {
			String[] tokens = line.split("\\s+");
			for (String token : tokens) {
				ntokens++;
				// if (token.equals(EOS))
				// break;
				int wid = getId(token);
				if (wid < 0) {
					continue;
				}
				entry_type type = getType(wid);
				if (type == entry_type.word && !discard(wid, (float) urd.sample())) {
					words.add(wid);
				}
				if (type == entry_type.label) {
					labels.add(wid - nwords_);
				}
				if (words.size() > MAX_LINE_SIZE && args.model != model_name.sup)
					break;
			}
		}
		return ntokens;
	}
	public void load(InputStream ifs) throws IOException {
		words_.clear();
		word2int_.clear();
		size_ = IOUtil.readInt(ifs);
		nwords_ = IOUtil.readInt(ifs);
		nlabels_ = IOUtil.readInt(ifs);
		ntokens_ = IOUtil.readLong(ifs);

		if (logger.isDebugEnabled()) {
			logger.debug("size_: " + size_);
			logger.debug("nwords_: " + nwords_);
			logger.debug("nlabels_: " + nlabels_);
			logger.debug("ntokens_: " + ntokens_);
		}

		for (int i = 0; i < size_; i++) {
			entry e = new entry();
			e.word = IOUtil.readString((DataInputStream) ifs);
			e.count = IOUtil.readLong(ifs);
			e.type = entry_type.fromValue(((DataInputStream) ifs).readByte() & 0xFF);
			words_.add(e);
			word2int_.put(find(e.word), i);

			if (logger.isDebugEnabled()) {
				logger.debug("e.word: " + e.word);
				logger.debug("e.count: " + e.count);
				logger.debug("e.type: " + e.type);
			}
		}
		initTableDiscard();
		initNgrams();
	}

}
