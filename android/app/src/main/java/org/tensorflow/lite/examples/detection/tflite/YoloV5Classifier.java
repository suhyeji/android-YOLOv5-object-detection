/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.examples.detection.MainActivity;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

public class YoloV5Classifier implements Classifier {
    public static YoloV5Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final boolean isQuantized,
            final int inputSize)
            throws IOException {
        final YoloV5Classifier d = new YoloV5Classifier();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
        }
        br.close();

        try {
            Interpreter.Options options = (new Interpreter.Options());
            options.setNumThreads(NUM_THREADS);
            if (isNNAPI) {
                d.nnapiDelegate = null;
                // Initialize interpreter with NNAPI delegate for Android Pie or above
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    d.nnapiDelegate = new NnApiDelegate();
                    options.addDelegate(d.nnapiDelegate);
                    options.setNumThreads(NUM_THREADS);
                    options.setUseNNAPI(true);
                }
            }
            if (isGPU) {
                GpuDelegate.Options gpu_options = new GpuDelegate.Options();
                gpu_options.setPrecisionLossAllowed(true); // It seems that the default is true
                gpu_options.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
                d.gpuDelegate = new GpuDelegate(gpu_options);
                options.addDelegate(d.gpuDelegate);
            }
            d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            d.tfLite = new Interpreter(d.tfliteModel, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.INPUT_SIZE = inputSize;
        d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.INPUT_SIZE * d.INPUT_SIZE];

        d.output_box = (int) ((Math.pow((inputSize / 32), 2) + Math.pow((inputSize / 16), 2) + Math.pow((inputSize / 8), 2)) * 3);
        if (d.isModelQuantized){
            Tensor inpten = d.tfLite.getInputTensor(0);
            d.inp_scale = inpten.quantizationParams().getScale();
            d.inp_zero_point = inpten.quantizationParams().getZeroPoint();
            Tensor oupten = d.tfLite.getOutputTensor(0);
            d.oup_scale = oupten.quantizationParams().getScale();
            d.oup_zero_point = oupten.quantizationParams().getZeroPoint();
        }

        int[] shape = d.tfLite.getOutputTensor(0).shape();
        int numClass = shape[shape.length - 1] - 5;
        d.numClass = numClass;
        d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * numBytesPerChannel);
        d.outData.order(ByteOrder.nativeOrder());
        return d;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }
    @Override
    public void enableStatLogging(final boolean logStats) {
    }

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
        tfLite.close();
        tfLite = null;
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnapiDelegate != null) {
            nnapiDelegate.close();
            nnapiDelegate = null;
        }
        tfliteModel = null;
    }

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
    }

    private void recreateInterpreter() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = new Interpreter(tfliteModel, tfliteOptions);
        }
    }

    public void useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = new GpuDelegate();
            tfliteOptions.addDelegate(gpuDelegate);
            recreateInterpreter();
        }
    }

    public void useCPU() {
        recreateInterpreter();
    }

    public void useNNAPI() {
        nnapiDelegate = new NnApiDelegate();
        tfliteOptions.addDelegate(nnapiDelegate);
        recreateInterpreter();
    }

    @Override
    public float getObjThresh() {
        return MainActivity.MINIMUM_CONFIDENCE_TF_OD_API;
    }

    private static final Logger LOGGER = new Logger();

    // Float model
    private final float IMAGE_MEAN = 0;

    private final float IMAGE_STD = 255.0f;

    //config yolo
    private int INPUT_SIZE = -1;

    private  int output_box;

    private static final float[] XYSCALE = new float[]{1.2f, 1.1f, 1.05f};

    private static final int NUM_BOXES_PER_BLOCK = 3;

    // Number of threads in the java app
    private static final int NUM_THREADS = 1;
    private static boolean isNNAPI = false;
    private static boolean isGPU = false;

    private boolean isModelQuantized;

    /** holds a gpu delegate */
    GpuDelegate gpuDelegate = null;
    /** holds an nnapi delegate */
    NnApiDelegate nnapiDelegate = null;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    // Config values.

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;

    private ByteBuffer imgData;
    private ByteBuffer outData;

    private Interpreter tfLite;
    private float inp_scale;
    private int inp_zero_point;
    private float oup_scale;
    private int oup_zero_point;
    private int numClass;
    private YoloV5Classifier() {
    }

    //non maximum suppression
    protected ArrayList<Recognition> nms(ArrayList<Recognition> list) {
        ArrayList<Recognition> nmsList = new ArrayList<Recognition>();

        for (int k = 0; k < labels.size(); k++) {
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    protected static final int BATCH_SIZE = 1;
    protected static final int PIXEL_SIZE = 3;

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    protected ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) ((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                    imgData.put((byte) (((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        return imgData;
    }

    public ArrayList<Recognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer_ = convertBitmapToByteBuffer(bitmap);

        Map<Integer, Object> outputMap = new HashMap<>();
        HashMap<String,String> nameMap = new HashMap<String,String>(){{
            put("10002", "해태썬키스트자몽소다350ML(PET)");
            put("10004", "웅진초록꿀매실180ML");
            put("10005", "웅진아침햇살500ML");
            put("10008", "코카씨그램플레인350ML");
            put("10009", "일화)맥콜500ML");
            put("10011", "조지아고티카빈티지블랙390ML");
            put("10013", "해태갈배사이다355ML");
            put("10014", "해태갈배사이다238ML");
            put("10015", "롯데핫식스자몽250ML");
            put("10016", "동아오츠카데미소다자몽250ML");
            put("10017", "롯데칸타타아메리카노200ML");
            put("10018", "롯데레쓰비175ML");
            put("10019", "코카콜라)킨사이다185ML");
            put("10021", "가야토마토농장180ML");
            put("10022", "코카콜라조지아고티카빈티지블랙270ML");
            put("10026", "롯데핑크퐁오렌지망고235ML");
            put("10027", "코카콜라오리지널300ML");
            put("10032", "롯데마운틴듀355ML");
            put("10033", "코카환타포도250ML");
            put("10034", "코카콜라제로250ML");
            put("10036", "코카콜라)코카콜라350ML");
            put("10037", "동아포카리스웨트500ML");
            put("10038", "롯데밀키스500ML");
            put("10040", "CJ)새싹보리410ML");
            put("10041", "롯데잘빠진하루우엉차500ML");
            put("10043", "광동)옥수수수염차500ML");
            put("10046", "동아오츠카마신다미네랄워터500ML");
            put("10047", "코카스프라이트500ML");
            put("10050", "롯데아이시스200ML");
            put("10052", "코카환타오렌지215ML");
            put("10060", "해태포키블루베리41G");
            put("10061", "카카오닙스차500ML");
            put("10063", "광동)힘찬하루헛개차1L");
            put("10066", "쟈뎅시그니처아메리카노스위트1.1L");
            put("10067", "쟈뎅시그니처아메리카노블랙1.1L");
            put("10068", "쟈뎅로얄헤이즐넛1.1L");
            put("10069", "쟈뎅시그니처까페라떼1.1L");
            put("10091", "꼬깔콘고소한맛72G");
            put("10092", "농심오징어집83G");
            put("10093", "농심매운새우깡90G");
            put("10094", "크라운)콘초66g");
            put("10095", "농심바나나킥75G");
            put("10101", "농심오징어짬뽕컵67G");
            put("10102", "농심신라면컵65G");
            put("10103", "오뚜기참깨라면(컵)");
            put("10104", "오뚜기)컵누들김치쌀국수34.8G");
            put("10111", "오뚜기컵누들매콤37.8G");
            put("10112", "농심)프링글스클래식53G");
            put("10113", "동서TOP심플리스무스스위트아메리카노240ML");
            put("10114", "동서TOP심플리스무스블랙240ML");
            put("10118", "레쓰비그란데라떼500ML");
            put("10122", "해태레몬에이드350ML");
            put("10123", "하이트진로)토닉워터300ML");
            put("10124", "롯데)칸타타콘트라베이스(스위트블랙)500ML");
            put("10125", "팔도뽀로로보리차220ML");
            put("10126", "델몬트허니애플시나몬280ML");
            put("10127", "웰치스그레이프500ml");
            put("10128", "웅진자연은토마토500ML");
            put("10130", "광동힘찬하루헛개차500ML");
            put("10131", "해태음료구론산바몬드오리지날액150ML");
            put("10132", "광동)발효홍삼100ml");
            put("10133", "CJ컨디션헛개100ML");
            put("10134", "광동)비타500(병)100ML");
            put("10135", "광동)비타500(병)180ML");
            put("10136", "현대미에로화이바210ML");
            put("10137", "해태갈아만든배238ML");
            put("10177", "롯데)오징어땅콩90G");
            put("10178", "농심칩포테토오리지날125G");
            put("10209", "해태)맛동산90G");
            put("10210", "오리온)포카칩오리지널66G");
            put("10235", "네모스낵매콤한맛");
            put("10247", "소와나무덴마크커피커피");
            put("10248", "소와나무덴마크초코초코우유");
            put("10249", "소와나무덴마크드링킹요구르트베리믹스");
            put("10251", "롯데오가닉유기농적포도보라핑크");
            put("10253", "파스퇴르위편한하루");
            put("10254", "밀크씨슬간편한닥터캡슐");
            put("10255", "빙그레사과에이드");
            put("10256", "매일바이오드링킹요커트플레인");
            put("10257", "프로틴고단백질요거트");
            put("10259", "소와나무덴마크드링킹요구르트파인애플");
            put("15026", "롯데빠다코코낫300G");
            put("15030", "크라운쿠크다스화이트토르테64G");
            put("15031", "퀘이커마시는오트밀50G");
            put("15032", "롯데)롯샌파인애플315G");
            put("15033", "롯데ABC초코쿠키152G");
            put("15034", "해태오예스콜드브루360G");
            put("15035", "롯데몽쉘오리지날생크림케이크384G");
            put("15036", "크라운쿠크다스케이크154G");
            put("15043", "해태롤리폴리초코196G");
            put("15044", "롯데야채크래커249G");
            put("15046", "동서리츠샌드위치크래커치즈96G");
            put("15047", "동서리츠샌드위치크래커레몬96G");
            put("15048", "롯데)롯샌화이트크림깜뜨315G");
            put("15049", "동서오레오초콜릿샌드위치쿠키100G");
            put("15050", "로투스베이커리즈코리아)로투스비스코프250G");
            put("15051", "청우아꾸뿔레102G");
            put("15052", "롯데마가렛트오리지널88G");
            put("15053", "삼양미트스파게티불닭볶음면110G");
            put("15054", "삼양짜장불닭볶음면105G");
            put("15055", "오뚜기떡라면140G");
            put("15076", "오리온다이제통밀28_194G");
            put("15123", "남양채움오렌지730ML");
            put("15124", "서울우유아침에주스자몽950ML");
            put("15125", "풀무원아임리얼케일키위700ML");
            put("15127", "서울우유아침에주스사과210ML");
            put("15133", "서울에프엔비아침에주스유기농토마토주스900ML");
            put("15142", "한국야쿠르트건강한간케어쿠퍼스프리미엄140ML");
            put("15146", "풀무원녹즙아임리얼스트로베리700ML");
            put("15175", "미가방유한회사)오레오씬즈바닐라무스84G");
            put("15176", "미가방유한회사)오레오씬즈티라미수84G");
            put("15182", "서울에프엔비)설빙인절미라떼250ML");
            put("15183", "매일유업)썬업(사과)200ML");
            put("15197", "서울우유강릉커피라떼250ML");
            put("15199", "동원커핑로카페봉봉");
            put("15200", "동원콩카페연유라떼");
            put("15201", "매일바리스타룰스로어슈거에스포레소라떼");
            put("15203", "동원다방커피카라멜");
            put("15204", "동원다방커피오리지널");
            put("15205", "남양프렌치카페로스터리블랙글레이즈드라떼");
            put("15206", "남양루카스나인시그니처더블샷라떼");
            put("15207", "남양루카스나인시그니처라떼");
            put("15212", "서울우유저지방나100_");
            put("15213", "서울우유(사각팩)");
            put("15216", "남양맛있는우유GT");
            put("15220", "서울에프앤비브루빈콜드브루");
            put("15221", "커피빈아메리카노");
            put("15338", "동서맥스윃하우스콜롬비아나마스터라떼500ML");
            put("15434", "크라운롱스220G");
            put("15622", "크라운스키피땅콩버터샌드위치쿠키288G");
            put("15623", "롯데명가찰떡파이375G");
            put("15654", "이마트피코크카스테라칩50G");
            put("15706", "해태구운감자슬림240G");
            put("15707", "크라운쿠크다스케이크140G");
            put("15708", "크라운빅파이딸기324G");
            put("15709", "오리온촉촉한초코칩240G");
            put("15823", "농심새우탕컵(소)67G");
            put("15825", "오뚜기컵누들팟타이쌀국수50G");
            put("15828", "오뚜기짜장볶이컵120G");
            put("15829", "오뚜기진라면매운맛컵110G");
            put("15830", "오뚜기진짜장컵115G");
            put("15831", "오뚜기진라면순한맛컵110G");
            put("15832", "오뚜기오동통면컵100G");
            put("15836", "오뚜기열라면컵105G");
            put("15838", "오뚜기육개장컵110G");
            put("15839", "삼양크림까르보불닭볶음면큰컵120G");
            put("15840", "농심사천짜파게티컵115G");
            put("15841", "팔도짬뽕왕뚜껑110G");
            put("15893", "해태블랙맛동산아몬드_헤이즐넛300G");
            put("15894", "해태흑당맛동산300G");
            put("15895", "해태맛동산375G");
            put("15906", "오뚜기컵누들얼큰쌀국수37.8G");
            put("15907", "오뚜기컵누들잔치쌀국수37.8G");
            put("15908", "남양우리아이처음먹는새우얌얌40G");
            put("15969", "크라운쿠크다스케이크사르르순크림154G");
            put("20003", "롯데2_복숭아350ML");
            put("20004", "롯데트로피카나스파클링사과355ML");
            put("20007", "웅진하늘보리500ML");
            put("20008", "웅진자연은석류180ML");
            put("20011", "미닛메이드사과175ml");
            put("20012", "롯데)레쓰비190ml");
            put("20016", "코카환타오렌지185ML");
            put("20021", "해태갈배사이다500ML");
            put("20022", "롯데펩시콜라600ML");
            put("20023", "롯데트레비플레인(PET)500ML");
            put("20025", "롯데펩시스트롱400ML");
            put("20026", "오케이에프)알로에베라킹500ml");
            put("20028", "해태)아침에사과500ML");
            put("20031", "코카콜라제로500ML");
            put("20033", "코카토레타500ML");
            put("20034", "롯데게토레이블루볼트600ML");
            put("20035", "코카씨그램라임350ML");
            put("20036", "코카씨그램라임450ML");
            put("20037", "롯데마운틴듀400ML");
            put("20038", "롯데칠성사이다로어슈거500ML");
            put("20039", "해태썬키스트스위티블루에이드350ML(PET)");
            put("20040", "GS리테일스파클링요구르트350ML");
            put("20041", "코카콜라)더탄산레몬450ML");
            put("20042", "롯데트로피카나스파클링사과400ML");
            put("20058", "에비앙naturalmineralwater");
            put("20061", "남양초코에몽250ML");
            put("20085", "오리온고소미80G");
            put("20089", "롯데제주사랑감귤사랑1.2L");
            put("20091", "롯데2_부족할때복숭아1.5L");
            put("20092", "미닛메이드스파클링청포도1.25L");
            put("20111", "오뚜기컵누들우동38.1G");
            put("20112", "오뚜기라면볶이120G");
            put("20113", "농심짜파게티큰사발123G");
            put("20114", "삼양)까르보불닭볶음면큰컵105G");
            put("20123", "해태에이스121G");
            put("20125", "프링글스오리지날110G");
            put("20126", "농심)프링글스클래식110G");
            put("20127", "프링글스치즈맛110G");
            put("20128", "농심)프링글스버터캬라멜110G");
            put("20129", "켈로그)프링글스사워크림_어니언");
            put("20138", "롯데)칠성사이다청귤");
            put("20139", "광동제약)비타500");
            put("20140", "그래미여명808140ML");
            put("20141", "해태오예스360G");
            put("20142", "오리온)초코파이(12입)468G");
            put("20144", "동서스타벅스프라프치노281ML");
            put("20148", "델몬트)그린매실180ML");
            put("20151", "맥심)티오페에스프레소");
            put("20152", "델몬트)토마토180ML");
            put("20153", "델몬트)골드망고180ML");
            put("20154", "조지아오리지날350ML");
            put("20155", "제주삼다수500ML");
            put("20156", "동서스타벅스모카281ML");
            put("20164", "해태)허니버터칩38G");
            put("20165", "삼양)사또밥갈릭버터맛52G");
            put("20166", "해태)구운대파70G");
            put("20167", "해태)오사쯔60G");
            put("20170", "해태)미니자유시간200G");
            put("20171", "농심알새우칩68G");
            put("20179", "오리온미쯔스낵팩84G");
            put("20180", "오리온)예감치즈그라탕2P64G");
            put("20182", "농심쫄병스낵안성탕면맛77G");
            put("20202", "오리온다이제씬84G");
            put("20211", "크라운)카라멜메이플콘74G");
            put("20220", "코카토레타1.5L");
            put("20234", "웅진빅토리아복숭아500ml");
            put("20240", "서울우유");
            put("20241", "빙그레쥬시쿨자두");
            put("20242", "동원쿨피스");
            put("20243", "덴마크민트라떼");
            put("20245", "덴마크드링킹요구르트자두");
            put("20246", "덴마크드링킹요구르트사과");
            put("20247", "덴마크드링킹요구르트청포도");
            put("20248", "덴마크바나나우유");
            put("20258", "제일훼미리키도크리미버터향크래커샌드위치");
            put("20323", "허쉬후퍼스340g");
            put("20325", "롯데초코파이420G");
            put("25006", "오리온예감오리지날12P192G");
            put("25010", "롯데애플쨈쿠키230G");
            put("25027", "크라운쵸코하임47G");
            put("25028", "크라운화이트하임47G");
            put("25029", "해태사브레252G");
            put("25033", "해태오예스쿠키앤크림360G");
            put("25034", "해태에이스436G");
            put("25035", "해태사브레315G");
            put("25042", "동서오레오초콜릿샌드위치쿠키300G");
            put("25043", "오리온다이제초코312G");
            put("25050", "크라운국희땅콩샌드155G");
            put("25051", "크라운초코하임142G");
            put("25052", "크라운참크래커56G");
            put("25053", "농심너구리큰사발면111G");
            put("25054", "오뚜기새우탕면110G");
            put("25055", "농심생생우동면276G");
            put("25129", "헬리코박터프로젝트윌150ML");
            put("25131", "풀무원아임리얼순수착즙오렌지700ML");
            put("25159", "남양야채채움퍼플190ML");
            put("25169", "푸르밀)프로바이오사과150ml");
            put("25172", "주식회사폴라리스)오레오웨이퍼룰초콜릿맛54G");
            put("25176", "삼아인터네셔날)오레오웨하스스틱(초코)");
            put("25179", "폴라리스)오레오웨이퍼룰바닐라맛54G");
            put("25183", "서울우유협동)아침에주스유기농포도쥬스900ML");
            put("25185", "매일유업)플로리다내추럴오렌지750ML");
            put("25186", "남양유업)채움포도730ML");
            put("25188", "풀무원식품)아임프룻제주한라봉감귤800ML");
            put("25189", "서울에프엔비)아침에주스사과950ML");
            put("25190", "풀무원식품)아임리얼토마토700ML");
            put("25191", "풀무원식품)아임리얼스무디마이스트로베리700ML");
            put("25193", "크라운제과)뽀또치즈타르트322G");
            put("25197", "롯데제과)도리토스나쵸치즈맛40G");
            put("25224", "동서식품)골든오레오샌드위치쿠키300G");
            put("25225", "크라운)초코칩쿠키미니270G");
            put("25328", "농심웰치스화이트그레이프맛355ML");
            put("25330", "동서맥스웰하우스콜롬비아나카페라떼240ML");
            put("25331", "동서맥스웰하우스콜롬비아나스위트아메리카노240ML");
            put("25332", "동서스타벅스더블샷에스플레소앤크림200ML");
            put("25333", "동서스타벅스파이크플레이스로스트블랙커피200ML");
            put("25334", "동서맥심에스프레소티오피마스터라떼200ML");
            put("25335", "남양프렌치카페마일드커피175ML");
            put("25336", "남양프렌치카페카라멜마끼아또175ML");
            put("25337", "코카)스프라이트215ML");
            put("25339", "코카)제로코카콜라215ML");
            put("25340", "코카)스프라이트1L");
            put("25341", "홈플러스시그니처사이다1.5L");
            put("25443", "동아오츠카데자와로얄밀크티500ML");
            put("25445", "롯데콘트라베이스티로스팅보리500ML");
            put("25446", "롯데콘트라베이스티로스팅그린티500ML");
            put("25447", "동원보성말차500ML");
            put("25469", "코카)아데스아몬드초콜릿190ML");
            put("25470", "코카)아데스아몬드오리지널190ML");
            put("25520", "위스트모구모구알로에베라큐브리치맛320ML");
            put("25521", "위스트모구모구파인애플맛320ML");
            put("25594", "오리온다이제미니320G");
            put("25595", "오리온다이제씬252G");
            put("25597", "오리온촉촉한초코칩320G");
            put("25598", "크라운쫀득초코칩240G");
            put("25600", "오리온초코칩쿠키256G");
            put("25601", "오리온예감볶은양파맛204G");
            put("25603", "오리온마켓오리얼크래커초코144G");
            put("25605", "해태얼려먹는초코만들기민트향36G");
            put("25609", "롯데디저뜨와233G");
            put("25617", "농심수미칩오리지널85G");
            put("25618", "농심수미칩어니언85G");
            put("25666", "매크로통상폴트초코앤헤이즐넛타르트150G");
            put("25668", "오리온예감치즈그라탕맛204G");
            put("25669", "이마트피코크대만파인에플케익(펑리수)270G");
            put("25835", "이멕스무역포테이토크리스프사워크림향75G");
            put("25836", "매크로통상르에트르다크플레이크트러플초콜릿175G");
            put("25853", "농심순한너구리컵63G");
            put("25877", "한국뉴초이스푸드아쌈밀크티300ML");
            put("25878", "주식회사위닝말리코코넛워터330ML");
            put("25967", "롯데칠성음료마스터토닉워터제로410ML");
            put("25968", "롯데칠성음료마스터토닉워터410ML");
            put("30007", "웰그린애플망고340ML");
            put("30010", "롯데핫식스355ML");
            put("30011", "화인바이오)지리산맑은샘물500ML");
            put("30012", "롯데델몬트포도100_400ML");
            put("30013", "코카콜라)고티카콜드브루카페라떼265ML");
            put("30014", "몬스터에너지울트라355ML");
            put("30015", "몬스터에너지그린355ML");
            put("30016", "웅진자연은제주감귤340ML");
            put("30017", "롯데밀키스250ML");
            put("30018", "롯데)칸타타스위트아메리카노275ML");
            put("30019", "롯데)칸타타프리미엄카페라떼200ML");
            put("30020", "롯데칸타타스위트아메리카노175ML");
            put("30021", "몬스터에너지파이프라인펀치355ML");
            put("30022", "코카콜라)미닛메이드망고175ML");
            put("30023", "코코팜포도238ML");
            put("30024", "롯데)칸타타프리미엄라떼175ML");
            put("30025", "동아오츠카화이브미니100ML");
            put("30026", "코카암바사345ML");
            put("30027", "코카조지아크래프트카페라떼470ML");
            put("30031", "빙그레따옴석류크랜베리235ML");
            put("30032", "따뜻한허니레몬_배280ML");
            put("30033", "코카콜라250ML");
            put("30036", "CJ헛개수340ML");
            put("30042", "광동)홍삼꿀D100ML");
            put("30044", "맥콜355ML");
            put("30045", "웅진)아침햇살(캔)180ML");
            put("30056", "델몬트콜드포도과즙100_1.89L");
            put("30057", "크라운쵸코하임284G");
            put("30058", "해태아이비270G");
            put("30059", "프레첼체다치즈맛85G");
            put("30060", "농심벌집핏자90G");
            put("30061", "농심포스틱84g");
            put("30062", "농심)에스키모밥60G");
            put("30063", "빙그레꽃게랑오리지널맛70G");
            put("30064", "크라운)꽃게랑불짬뽕맛70G");
            put("30066", "농심꿀꽈배기90G");
            put("30067", "오리온)포카칩양파66G");
            put("30070", "해태)구운양파70G");
            put("30078", "삼양)치즈불닭볶음면큰컵105G");
            put("30079", "오뚜기스파게티컵120G");
            put("30087", "해태)크림웨하스Original50G");
            put("30088", "롯데칙촉90G");
            put("30090", "농심닭다리핫숯불바베큐66G");
            put("30091", "해태)초코틴틴88G");
            put("30092", "롯데)엄마손파이127G");
            put("30093", "롯데)ABC초코쿠키50G");
            put("30094", "오리온)고소미2P70G");
            put("30095", "크라운)참아이엔지치즈크림135G");
            put("30097", "크라운)참아이엔지치즈레몬135G");
            put("30098", "롯데)몽쉘카카오생크림케이크192G");
            put("30099", "롯데)빠다코코낫100G");
            put("30100", "롯데)제크오리지날100G");
            put("30106", "해태)뉴얼려먹는초코만들기36G");
            put("30120", "롯데)꼬깔콘매콤달콤72G");
            put("30133", "포카칩오리지널110G");
            put("30137", "롯데)에어베이크드포테이토오리지날85G");
            put("30138", "오리온다이제초코225G");
            put("30140", "농심인디안밥83G");
            put("30152", "토하토)크레용신찬20G");
            put("30166", "롯데)치토스후라이드_양념치킨맛80G");
            put("30189", "웅진하늘보리1.5L");
            put("30203", "웅진티즐유자그린티500ml");
            put("30210", "매일피크닉청포도");
            put("30211", "매일아몬드브리즈뉴트리플러스프로틴");
            put("30212", "매일썬업그린(과일야채샐러드)");
            put("30215", "남양이오트웬티즈");
            put("30217", "남양맛있는우유GT");
            put("30218", "소와나무쿨피스오리지널복숭아");
            put("30219", "소와나무쿨피스오리지널파인애플");
            put("30226", "CJ컨디션레이디100ML");
            put("30291", "프링글스오리지날53G");
            put("30292", "프링글스양파맛53G");
            put("35013", "씨그램애플리프레시350ML");
            put("35026", "크라운쿠크다스케이크사르르순크림77G");
            put("35030", "롯데마가렛트오리지널396G");
            put("35032", "롯데하비스트피넛버터샌드91G");
            put("35035", "팔도왕뚜껑110G");
            put("35039", "롯데ABC초코쿠키쿠키앤크림43G");
            put("35040", "청우쫀득초코칩240G");
            put("35043", "오리온후레쉬베리복숭아336G");
            put("35044", "해태후렌치파이딸기256G");
            put("35045", "해태후렌치파이사과256G");
            put("35047", "동서딸기크림오레오샌드위치쿠키300G");
            put("35049", "삼아꼬마웨하스바닐라맛80G");
            put("35050", "해태홈런볼티라미수146G");
            put("35117", "매일유업)까망베르자연치즈100G");
            put("35119", "풀무원식품)아임리얼스트로베리190ML");
            put("35120", "풀무원식품)아임리얼순수착즙오렌지190ML");
            put("35121", "풀무원식품)아임리얼스무디아이스트로베리190ML");
            put("35122", "풀무원식품)아임프룻청송사과215ML");
            put("35124", "푸르밀)달고나라떼250ML");
            put("35125", "거창공장)아침에주스950ML");
            put("35130", "남양)야채체움녹황190ML");
            put("35157", "서울우유)아침에주스포도950ML");
            put("35161", "농심)신라면큰사발면114G");
            put("35178", "풀무원)액티비아사과130ML");
            put("35179", "풀무원)약티비아딸기130ML");
            put("35203", "매일유업)바리스타룰스플라넬드립라떼325ML");
            put("35204", "푸르밀)생초콜릿라떼250ML");
            put("35205", "엠즈씨드)폴바셋콜드브루보틀300ML");
            put("35206", "남양유업)채움포도210ML");
            put("35207", "남양유업)초코에몽180ML");
            put("35209", "남양)채움오렌지210ML");
            put("35215", "한국야쿠르트)헬리코박터프로젝트윌저지방150ML");
            put("35216", "상하목장)유기농딸기우유125ML");
            put("35218", "삼양)불닭볶음탕면120G");
            put("35219", "삼양)쫄볶이불닭볶음면120G");
            put("35220", "삼양)삼양라면매운맛110G");
            put("35225", "농심)신라면블랙101G");
            put("35228", "농심)감자면92G");
            put("35389", "코카)아데스아몬드오리지널");
            put("35418", "롯데)델몬트콜드복숭아과즙100_");
            put("35419", "롯데)델몬트콜드사과주스");
            put("35600", "농심백산수330ML");
            put("35609", "아쿠아리어스골드메달스파클링애플주스296ML");
            put("35637", "이마트)프링글스사워크림앤어니언");
            put("35638", "이마트)프링글스오리지널");
            put("35645", "델리팜)로아커웨하스더블초코");
            put("35646", "델리팜)로아커웨하스코코아_밀크");
            put("35685", "삼양패키징)맥심TOP심플리스무스로스티라떼");
            put("35688", "한국음료)조지아고티카빈티지라떼");
            put("35689", "동서식품)맥심TOP마스터라떼");
            put("35690", "금강B_F)칸타타아메리카노");
            put("35695", "리엘산업)코코리치위드나타드코코");
            put("35699", "농심)파워오투아이스베리향");
            put("35700", "농심)파워오투애플키워향");
            put("35702", "농심)파워오투오렌지레몬향");
            put("35703", "삼양패키징)맥심TOP심플리스무스라떼");
            put("35704", "농심)오이오차녹차");
            put("35705", "동원)바이오티피치핑크");
            put("35707", "삼양패키징)티즐피치우롱티");
            put("35708", "ATG컴퍼니)푸룬주스");
            put("35710", "건강한사람들)과일사이다딸기키위");
            put("35711", "웅진식품)이온더핏");
            put("35714", "동서식품)스타벅스프라푸치노(카라멜향)");
            put("35742", "델리팜)로아커웨하스다크초콜릿");
            put("35855", "에이플네이처)퍼펙트파워쉐이크바닐라향");
            put("35856", "에이플네이처)퍼펙트파워쉐이크초코바나나맛");
            put("35857", "서울에프엔비)마시는식사마일드카카오");
            put("35858", "서울에프엔비)마시는식사마일드라떼");
            put("40002", "롯데레쓰비마일드커피200ml");
            put("40003", "동아오츠카오란씨파인애플180ml");
            put("40004", "해태봉봉포도238ML");
            put("40005", "롯데비타파워180ml");
            put("40007", "롯데마운틴듀250ML");
            put("40008", "코카콜라조지아카페마일드175ml");
            put("40009", "코카콜라조지아카페스위트블랙175ml");
            put("40010", "해태갈배사이다210ML");
            put("40011", "델몬트알로에180ml");
            put("40012", "롯데)2_부족할때아쿠아500ML");
            put("40014", "농심웰치스포도355ml");
            put("40015", "해태갈아만든배340ML");
            put("40017", "현대미에로화이바100ML");
            put("40019", "에이드요구르트340ML");
            put("40021", "코카파워에이드MB600ML");
            put("40022", "가야알로에농장180ML");
            put("40023", "롯데레쓰비모카라떼175ML");
            put("40024", "롯데트레비라임300ML");
            put("40025", "롯데델몬트망고400ML");
            put("40026", "롯데)칸타타콘트라베이스콜드브루라떼500ML");
            put("40027", "롯데)립톤밀크티240ML");
            put("40028", "롯데핫식스더킹파워355ML");
            put("40029", "빅토리아라임350ML");
            put("40030", "코카파워에이드MB240ML");
            put("40031", "롯데레쓰비연유커피베트남240ML");
            put("40032", "롯데)레쓰비아시아트립쏠트커피240ML");
            put("40033", "롯데)펩시콜라210ML");
            put("40036", "오란씨깔라만시250ML");
            put("40037", "코코팜망고코넛340ML");
            put("40038", "롯데칸타타프리미엄라떼275ML");
            put("40039", "칸타타프리미엄카페라떼390ML");
            put("40040", "코카파워에이드MB520ML");
            put("40041", "롯데)데일리C레몬1000C_파우치230ml");
            put("40042", "해태)썬키스트훼미리포도180ML");
            put("40043", "롯데실론티240ML");
            put("40044", "해태파인애플봉봉240ML");
            put("40045", "롯데게토레이240ML");
            put("40046", "코카콜라캐나다드라이클립소다250ML");
            put("40047", "해태)복숭아봉봉340ML");
            put("40048", "롯데)레쓰비카페타임라떼240ML");
            put("40049", "코카환타오렌지250ML");
            put("40050", "코카콜라)조지아고티카빈티지라떼270ML");
            put("40051", "심플러스바른샘물300ML");
            put("40052", "코카콜라)코카콜라오리지날테이스트250ml");
            put("40053", "현대미에로화이바350ML");
            put("40054", "코카조지아고티카스위트아메리카노270ML");
            put("40055", "아이시스8.0300ml");
            put("40057", "해태)코코팜복숭아300ML");
            put("40058", "해태강원평창수500ML");
            put("40059", "광동우엉차500ml");
            put("40061", "푸르밀검은콩우유300ml");
            put("40065", "롯데트레비플레인(CAN)190ML");
            put("40068", "웅진옥수수수염차180ML");
            put("40070", "롯데펩시콜라1.25L");
            put("40072", "롯데트레비플레인(PET)1.2L");
            put("40074", "웅진자연은알로에1.5L");
            put("40075", "롯데볼빅1.5L");
            put("40076", "한국청정음료몽베스트2L");
            put("40077", "롯데게토레이블루볼트1.5L");
            put("40078", "해태코코팜피치핑크복숭아1.5L");
            put("40088", "오리온썬핫스파이시맛80G");
            put("40089", "오리온꼬북칩80G");
            put("40090", "롯데치토스스모키바베큐맛88G");
            put("40091", "꼬깔콘군옥수수맛72G");
            put("40092", "농심감자깡75G");
            put("40095", "농심새우깡90G");
            put("40103", "오뚜기진진짜라120G");
            put("40104", "오뚜기크림진짬뽕105G");
            put("40105", "롯데)초코빼빼로54G");
            put("40106", "롯데)아몬드빼빼로37G");
            put("40109", "해태칼로리바란스76G");
            put("40110", "롯데)누드초코빼빼로50G");
            put("40143", "오리온)치킨팝81G");
            put("40144", "삼양별뽀빠이72G");
            put("40145", "오리온)무뚝뚝감자칩60G");
            put("40146", "롯데쌀로별오리지널78G");
            put("40147", "오리온오감자양념바베큐맛50G");
            put("40148", "농심수미칩어니언55G");
            put("40150", "삼양짱구115g");
            put("40153", "오)포카칩어니언맛110g");
            put("40154", "오리온태양의맛썬64g");
            put("40155", "해태)빠새60G");
            put("40156", "해태맛동산165g");
            put("40183", "농심)포테토칩육개장사발면맛60G");
            put("40223", "웅진빅토리아청포도500ml");
            put("40227", "빙그레요플레프로틴고단백질요거트");
            put("40228", "롯데파스퇴르쾌변골드키위");
            put("40231", "빙그레아카페라가페라떼");
            put("40232", "매일바이오드링킹요거트블루베리");
            put("40233", "빙그레쥬시쿨복숭아");
            put("40234", "서울우유살롱밀크티");
            put("40235", "서울에프엔비쥬시쿨청포도에이드");
            put("40236", "서울에프엔비쥬시쿨금귤_감귤에이드");
            put("40246", "롯데씨리얼초코");
            put("40314", "롯데빈츠76G");
            put("45036", "롯데찰떡파이인절미300G");
            put("45038", "크라운화이트하임284G");
            put("45045", "해태아이비155G");
            put("45047", "농심신라면건면");
            put("45049", "크라운딸기파이");
            put("45050", "오리온카스타드");
            put("45051", "삼하꼬마웨하스");
            put("45052", "델리팜로아커웨하스밀크바닐라");
            put("45053", "델리팜로아커웨하스바닐라");
            put("45125", "풀무원)풀무원다른액티비아포도130ML");
            put("45126", "풀무원)풀무원아임그린보리새싹150ML");
            put("45127", "매일유업)썬업과일야채샐러드레드200ML");
            put("45128", "풀무원)아임프릇제주한라봉감귤215ML");
            put("45131", "피스퇴르)안나오면쳐들어간다쾌변포도150ML");
            put("45132", "풀무원)다논액티비아화이트130ML");
            put("45134", "매일유업)매일썬업오렌지100_200ML");
            put("45136", "한국야쿠르트)하루야채퍼플200ML");
            put("45137", "풀무원)아임그린아스파라거스150ML");
            put("45161", "오리온)초코칩쿠키90G");
            put("45167", "농심켈로그)프링글스블로네제스파게티맛110G");
            put("45168", "농심켈로그)프링글스사워크림앤어니언100G");
            put("45169", "농심)사리곰탕컵61G");
            put("45209", "매일유업)아메리카노싱글오리진코스타리카330ML");
            put("45210", "서울에프엔비)볼드에스프레소라떼300ML");
            put("45211", "서울에프엔비)트루에스프레소블랙300ML");
            put("45212", "동원F_B)달달하고부드러운흑당크림라떼300ML");
            put("45213", "남양유업)프렌치카페로스터리에스프레소라떼250ML");
            put("45214", "남양유업)루카스나인시그니처아메리카노250ML");
            put("45215", "동원F_B)콩카페코코넛라떼250ML");
            put("45219", "오리온)예감32G");
            put("45220", "오리온)예감치즈그라탕맛32G");
            put("45227", "롯데제과)꼬깔콘고소한맛42G");
            put("45237", "서주제과)서주허쉬웨하스미니100G");
            put("45238", "서주제과)서주허쉬민트훼하스미니100G");
            put("45269", "매일유업)유기농코코아우유125ML");
            put("45270", "매일유업)유기농바나나우유125ML");
            put("45271", "매일유업)유기농우유125ML");
            put("45293", "삼양패키징)콜롬비아나마스터블랙500ML");
            put("45298", "코카-콜라음료)조지아고티카콜드브루아메리카노265ML");
            put("45306", "동서식품)파이크플레이스로스트블랙커피275ML");
            put("45397", "한국쥬맥스)모구모구복숭아향");
            put("45398", "한국쥬맥스)모구모구리치맛");
            put("45400", "이콤)황성주약콩두유");
            put("45401", "롯데코카콜라");
            put("45505", "롯데제과)빼빼로돼지바");
            put("45531", "농심)에너지바K크런치넛(레드베리)");
            put("45663", "한국청정음료)트루워터");
            put("45664", "광동)제주삼다수");
            put("45665", "제이크리에이션)제주트루워터");
            put("45762", "하임상사)한입자색찐고구마");
            put("45792", "서주제과)서주허쉬초코웨하스");
            put("45793", "서주제과)서주허쉬민트웨하스");
            put("50003", "바리스타룰스)콜드브루블랙325ML");
            put("50004", "롯데)미린다오렌지355ML");
            put("50005", "롯데비타파워100ml");
            put("50011", "롯데트로피카나스파클링포도355ML");
            put("50012", "일화맥콜250ML");
            put("50013", "일화초정탄산수190ML");
            put("50014", "롯데칸타타오리지날원두커피390ml");
            put("50015", "오케이에프요거상큼코코앤복숭아340ML");
            put("50017", "코카파워에이드MB355ML");
            put("50018", "롯데사각사각꿀배340ML");
            put("50019", "동아포카리스웨트(캔)340ML");
            put("50020", "오케이에프미닛메이드알로에175ML");
            put("50021", "nutrlon_taste수박소다350ML");
            put("50022", "LOTTE레쓰비마일드커피150ML");
            put("50024", "바이오휘오제주500ML");
            put("50026", "코카콜라190ML");
            put("50027", "코카조지아커피오리지널240ML");
            put("50028", "코카)미닛메이드조이오렌지175ML");
            put("50029", "컨디션CEO150ML");
            put("50030", "롯데칸타타에스프레소블랙175ML");
            put("50061", "오리온스윙칩볶음고추장60g");
            put("50062", "농심자갈치90G");
            put("50063", "도리토스갈비천왕치킨맛172G");
            put("50070", "CJ인삼한뿌리120ML");
            put("50072", "오리온)예감오리지날2P64G");
            put("50073", "롯데상큼달콤롯데샌드화이트크림깜뜨105G");
            put("50074", "동서)오레오솔티드카라멜100G");
            put("50075", "오레오레드벨벳샌드위치쿠키94G");
            put("50076", "동서)오레오씬즈라즈베리무스84G");
            put("50077", "동서오레오초콜릿크림");
            put("50078", "롯데롯샌상큼달콤파인애플");
            put("50080", "코카파워에이드퍼플스톰600ML");
            put("50087", "박카스F120ML");
            put("50089", "롯데)칙촉티라미수90G");
            put("50090", "크라운)뽀또치즈타르트161G");
            put("50095", "롯데카스타드138G");
            put("50097", "해태)구운고구마27G");
            put("50099", "해태사브레84G");
            put("50100", "해태)자가비대파_로메스코소스45G");
            put("50117", "크라운)콘칩(군옥수수)70G");
            put("50129", "동서골든오레오100G");
            put("50145", "크라운롱화이트하임47G");
            put("50163", "팔도비락식혜1.8L");
            put("50176", "게메즈에낙");
            put("50178", "웅진)아침햇살1.5L");
            put("50182", "롯데델몬트포도드링크4입190ml");
            put("50186", "빙그레아카페라사이즈업아메리카노");
            put("50187", "서울우유딸기");
            put("50189", "애경말랑카우를좋아하는말랑이버블핸드워시");
            put("50190", "서울우유초콜릿");
            put("50191", "서울우유커피");
            put("50192", "소와나무덴마크딸기딸기우유");
            put("50194", "빙그레아카페라카라멜마끼아또");
            put("50196", "매일바이오드링킹요거트매실푸룬");
            put("50197", "매일바이오백도요거트");
            put("50198", "매일바이오블루베리요거트");
            put("50199", "크라운버터와플");
            put("50203", "오레오미니오레오딸기");
            put("50205", "광동다복쌍화150ML");
            put("50261", "오리온무뚝뚝감자칩124G");
            put("55023", "농심신라면두부김치94G");
            put("55029", "롯데빈츠204G");
            put("55030", "크라운국희땅콩샌드372G");
            put("55031", "롯데칙촉2번들(1682입)");
            put("55032", "크라운초코하임284G");
            put("55041", "농심짜왕큰사발면");
            put("55045", "오리온쫀득쫀득참붕어빵");
            put("55046", "오리온다이제초코");
            put("55050", "롯데빠다코코낫");
            put("55112", "아침에주스제주감귤");
            put("55116", "풀무원아임그린");
            put("55117", "풀무원액티비아블루베리");
            put("55118", "풀무원액티비아플레인");
            put("55120", "한국야쿠르트하루야채뽀로로");
            put("55121", "풀무원아임프룻청송사과");
            put("55125", "풀무원아임그린민트");
            put("55126", "서울우유아침에주스제주감귤");
            put("55128", "알미체리페퍼위드스파이스크림치즈");
            put("55180", "서울우유딸기");
            put("55181", "남양우유GT찐_하고달달한딸기");
            put("55183", "동서스타벅스카페라떼");
            put("55184", "매일소화가잘되는우유오리지널락토프리");
            put("55187", "매일소화가잘되는우유저지방2_락토프리");
            put("55189", "네슬레네스퀵초콜릿만드링크");
            put("55190", "매일아몬드브리즈오리지널");
            put("55191", "매일상하목장유기농우유저지방");
            put("55193", "남양우유GT찐_하고달달한초코");
            put("55196", "한국야쿠르트헬리코박터프로젝트윌저지방");
            put("55198", "매일상하목장유기농우유");
            put("55201", "한국야구르트핫브루바닐라라떼");
            put("55202", "남양채움포도");
            put("55207", "매일아몬드브리즈언스위트(무당)");
            put("55208", "남양맛있는우유GT소화잘되는배안아픈우유");
            put("55537", "오리온초코파송이2개가한묶음");
            put("55691", "이마트티타임비스킷벨기에산");
            put("55710", "젠니혼주류치어스다이긴죠");
            put("55717", "델몬트망고드링크(1개)");
            put("55738", "광동제약)광동우롱차");
            put("55741", "동서식품)맥스웰하우스콜롬비아나마스터스위트블랙");
            put("55744", "동서식품)스타벅스프라푸치노");
            put("55746", "에이치케이이노엔)컨디션헛개수이엑스");
            put("55752", "동서식품)스타벅스파이크플레이스로스트커피");
            put("55753", "동서식품)스타벅스시그니처초콜렛");
            put("55754", "동서식품)스타벅스브렉퍼스트블렌드블랙커피");
            put("55821", "이멕스무역)포테이토크리스프바비큐맛5개입");
            put("55822", "이멕스무역)포테이토크리스프스위트콘맛5개입");
            put("55823", "이멕스무역)스위트포테이토크리스프5개입");
            put("56042", "동서식품)마일드스위트오레오초콜릿샌드위치쿠키");
            put("60001", "롯데칠성사이다로어슈거250ml");
            put("60002", "코카스프라이트알루미늄보틀250ML");
            put("60008", "일화천연사이다250ML");
            put("60009", "롯데펩시콜라250ML");
            put("60012", "빅토리아(레몬)350ML");
            put("60013", "오케이에프빅토리아플레인350ML");
            put("60015", "롯데트로피카나스파클링망고355ML");
            put("60018", "빅토리아자몽350ml");
            put("60034", "롯데이프로부족할때복숭아240ML");
            put("60036", "롯데델몬트망고240ML");
            put("60037", "코카환타파인애플250ML");
            put("60038", "동아데자와240ML");
            put("60039", "일화)맥콜190ML");
            put("60042", "롯데핫식스더킹펀치355ML");
            put("60043", "롯데)펩시콜라160ML");
            put("60044", "심플러스)바른샘물500ML");
            put("60045", "웅진누룽지500ML");
            put("60046", "웅진광명찾은결명자차500ML");
            put("60047", "코카암바사500ML");
            put("60048", "농심백산수500ML");
            put("60049", "삼양패키징)복숭아워터410ML");
            put("60050", "복숭아녹차340ML");
            put("60051", "세븐일레븐)깊은산속옹달샘물");
            put("60052", "롯데플러스펄프오렌지스파클링에이드500ML");
            put("60053", "동서제티초코175ML");
            put("60054", "웅진)홍삼꿀d100ml");
            put("60055", "웅진가을대추280ml");
            put("60057", "롯데)칸타타콘트라베이스블랙400ML");
            put("60058", "일화)맥콜1.25l");
            put("60060", "코카콜라단짠커피240ML");
            put("60090", "크라운초코하임47G");
            put("60091", "농심사리곰탕큰사발111G");
            put("60094", "삼양라면110g");
            put("60095", "오뚜기)쇠고기미역국라면100G");
            put("60097", "농심프링글스핫앤스파이시53G");
            put("60108", "농심새우탕큰사발115G");
            put("60110", "오뚜기스낵면컵");
            put("60111", "농심김치사발면86G");
            put("60113", "오뚜기)컵누들베트남쌀국수컵47G");
            put("60114", "농심짜파게티범벅70G");
            put("60115", "오뚜기진라면컵순65G");
            put("60116", "농심)앵그리짜파구리큰사발108G");
            put("60118", "오뚜기진라면매운맛65G(작은용기)");
            put("60119", "오뚜기육개장용기면86G");
            put("60120", "삼양큰컵불닭볶음면105G");
            put("60121", "오뚜기튀김우동컵110G");
            put("60130", "동서오레오딸기크림100G");
            put("60131", "프링글스마요치즈맛110G");
            put("60147", "오리온닥터유다이제194G");
            put("60174", "오리온참붕어빵348G");
            put("60193", "웅진빅토리아파인애플500ml");
            put("60199", "빙그레요구르트");
            put("60201", "동원쿨피스파인애플");
            put("60202", "닥터캡슐프로텍트사과");
            put("60203", "닥터캡슐프로텍트베리믹스");
            put("60204", "매일우유저지방");
            put("60206", "매일썬업과일야채샐러드녹황");
            put("60209", "소와나무덴마크드링킹요구르트복숭아");
            put("60210", "매일썬업과일야채샐러드레드");
            put("60228", "연세마카다미아초코우유");
            put("60282", "심플러스오리지날감자칩치즈");
            put("65029", "오리온오뜨쇼콜라300G");
            put("65031", "오리온오뜨치즈288G");
            put("65038", "미성패밀리허쉬초코크림샌드위치쿠키");
            put("65044", "농심보글보글부대찌개큰사발면");
            put("65045", "농심볶음너구리큰사발면");
            put("65047", "오리온초초칩쿠키미니");
            put("65050", "동서초콜릿크림오레오");
            put("65119", "서울우유듀오안저지방플레인");
            put("65120", "서울우유듀오안오리지널");
            put("65125", "서울우유아침에주스포도");
            put("65191", "해태아이비");
            put("65199", "켈로그프링글스찹스테이크");
            put("65200", "켈로그프링글스크리미쉬림프");
            put("65457", "해태에이스(10개입)");
            put("65470", "해태얼초집만들기");
            put("65619", "청우쫀득쫀득초코파이찰떡10개입");
            put("65703", "농심)백산수");
            put("65711", "롯데칠성음료)에비앙천연광천수");
            put("65723", "롯데칙촉오리지날");
            put("65727", "롯데몽쉘카카오생크림케이크12봉");
            put("65852", "남양)몸이가벼워지는시간17차오리진");
            put("65858", "오리온)크런치케이준눈을감자");
            put("65915", "보람비티)미니카안녕자두야바삭바삭맛있는초코과자초코펀초코맛");
            put("65916", "유한회사아이디어원)쥐방울멜티키스밀크초코볼");
            put("66188", "에이스엠엔티)미니오레오");
            put("66360", "그래미여명1004천사의행복140ML");
            put("70002", "롯데트로피카나스파클링복숭아355ML");
            put("70034", "가야토마토농장500ML");
            put("70035", "코카태양의마테차500ML");
            put("70036", "바리스타룰스)마다가스카르바닐라빈라떼325ml");
            put("70040", "동아포카리스웨트620ML");
            put("70042", "빅토리아(자몽)500ML");
            put("70043", "롯데에비앙500ML");
            put("70044", "하이트진로)토닉워터깔라만시300ML");
            put("70046", "코카환타파인애플600ML");
            put("70047", "가야알로에농장500ML");
            put("70051", "광동야관문야왕500ML");
            put("70054", "웅진하늘보리325ml");
            put("70056", "롯데칠성사이다245ML");
            put("70061", "코카스프라이트355ML");
            put("70065", "롯데델몬트콜드수박주스1L");
            put("70066", "롯데델몬트콜드포도과즙100_1L");
            put("70067", "푸드웰레몬비타1000_150ML");
            put("70076", "코카파워에이드퍼플스톰1.5L");
            put("70078", "일화)맥콜1.5L");
            put("70080", "동아데미소다애플1.5L");
            put("70082", "해태)맛동산300G");
            put("70083", "오리온)눈을감자113G");
            put("70086", "크라운)못말리는신짱120G");
            put("70089", "허쉬)초코크림샌드위치쿠키100G");
            put("70093", "해태구운감자27G");
            put("70096", "농심닭다리후라이드66G");
            put("70097", "청우초코파이찰떡");
            put("70099", "해태)자가비짭짤한맛45G");
            put("70102", "롯데)에어베이크드포테이토사워크림어니언맛70G");
            put("70106", "크라운)롱스132G");
            put("70107", "크라운산도딸기크림치즈161G");
            put("70125", "농심)쫄병스낵매콤한맛82G");
            put("70126", "오뚜기뿌셔뿌셔양념치킨맛90G");
            put("70143", "농심)프링글스블랙페퍼크랩110G");
            put("70160", "농심)쫄병스낵짜파게티맛77G");
            put("70210", "매일아몬드브리즈언스위트");
            put("70211", "파스퇴르야채농장");
            put("70212", "파스퇴르야채농장ABC");
            put("70213", "매일썬업저과즙콜라겐플랜");
            put("70214", "서울우유바나나");
            put("70215", "서울우유");
            put("70216", "매일바이오드링킹요거트스트로베리");
            put("70217", "매일썬업과일야채샐러드퍼플");
            put("70219", "한국야구르트뽀짝뽀짝포도사과");
            put("70220", "빙그레아카페라잇츠라떼리치연유");
            put("75009", "크라운)쿠크다스비엔나커피");
            put("75048", "농심튀김우동컵62G");
            put("80007", "롯데푸드)오가닉유기농레드비트_배_토마토125ML");
            put("80008", "파스퇴르오가닉유기농사과_당근125ML");
            put("80010", "롯데칠성사이다190ML");
            put("80011", "코카콜라)미닛메이드벚꽃_사과175ML");
            put("80012", "해태)코코팜화이트요구르트240");
            put("80014", "롯데레쓰비마일드커피240ML");
            put("80015", "롯데레쓰비카페타임스위트아메리카노240ML");
            put("80017", "롯데트로피카나스파클링오렌지355ML");
            put("80018", "롯데밀키스340ML");
            put("80022", "서울아침에주스자몽210ML");
            put("80023", "빙그레따옴천혜향한라봉청귤235ML");
            put("80024", "코카환타오렌지미니300ML");
            put("80026", "레몬녹차340ML");
            put("80028", "롯데트레비플레인(PET)300ML");
            put("80029", "롯데핑크퐁포도사과235ML");
            put("80030", "롯데델몬트알로에400ML");
            put("80031", "롯데델몬트오렌지100_400ML");
            put("80035", "해태)태양의식후비법W차500ML");
            put("80036", "빅토리아(플레인)500ML");
            put("80037", "웅진초록매실500ML");
            put("80038", "롯데이프로부족할때복숭아500ML");
            put("80039", "롯데트레비자몽(PET)500ML");
            put("80040", "롯데트레비라임(PET)500ML");
            put("80041", "롯데)립톤아이스티복숭아500ML");
            put("80042", "롯데GS보리차");
            put("80043", "롯데황금보리500ML");
            put("80045", "롯데트로피카나스파클링복숭아400ML");
            put("80046", "롯데게토레이600ML");
            put("80047", "롯데아이시스8.01L");
            put("80082", "롯데)초코쿠키빼빼로37G");
            put("80083", "롯데)빼빼로누드크림치즈46G");
            put("80084", "롯데)화이트쿠키빼빼로37G");
            put("80085", "롯데)빼빼로더슬림45G");
            put("80086", "롯데)크런키빼빼로39G");
            put("80087", "해태포키딸기41G");
            put("80088", "해태)포키극세44G");
            put("80092", "해태)포키46G");
            put("80093", "해태구운감자108G");
            put("80094", "크라운쿠크다스커피72G");
            put("80098", "해태버터링86G");
            put("80099", "오리온고래밥볶음46G");
            put("80100", "크라운쿠크다스화이트128G");
            put("80103", "롯데마가렛트176G");
            put("80104", "오리온)배배80G");
            put("80105", "롯데몽쉘크림192G");
            put("80106", "해태후렌치파이사과192G");
            put("80107", "해태후렌치파이딸기192G");
            put("80108", "오리온)눈을감자M56G");
            put("80117", "프링글스핫스파이시110G");
            put("80160", "롯데펩시콜라1.5L");
            put("80168", "코카스프라이트1.5L");
            put("80182", "오리온통크초코");
            put("80183", "롯데파스퇴르쾌변포도");
            put("80184", "서울우유딸기");
            put("80185", "매일허쉬초콜릿드링크쿠키앤크림");
            put("80186", "남양맛있는우유GT");
            put("80187", "매일우유속에딸기과즙");
            put("80189", "서울우유커피");
            put("80190", "빙그레아카펠라바닐라라떼");
            put("80191", "생생초스위트몬스터워터젤리사과");
            put("80205", "서울스페셜티까페라떼다크");
            put("80206", "연세달고나커피우유");
            put("80264", "ABC초코쿠키130G");
            put("85031", "보해양조부라더소다밀키소다맛750ML");
            put("90001", "롯데)2_아쿠아수분_미네랄240ML");
            put("90002", "해태포도봉봉340ML");
            put("90003", "롯데)쌕쌕오렌지238ML");
            put("90004", "코카)환타레몬355ML");
            put("90005", "팔도비락수정과238ML");
            put("90007", "썬키스트훼미리사과180ML");
            put("90008", "칸타타콜드브루블랙275ML");
            put("90009", "동아오츠카오란씨오렌지250ML");
            put("90010", "코카파워에이드MB355ML(PET)");
            put("90011", "롯데)사랑초톡톡스파클링파인애플330ML");
            put("90012", "코카환타오렌지600ML");
            put("90015", "코카조지아고티카콜드브루스위트아메리카노265ML");
            put("90016", "롯데자몽워터500ML");
            put("90017", "롯데트레비레몬(PET)500ML");
            put("90018", "웅진코코몽유기농하늘보리200ML");
            put("90019", "롯데칸타타땅콩크림라떼275ML");
            put("90020", "동아데미소다애플250ML");
            put("90021", "칸타타콜드브루라떼275ML");
            put("90022", "광동탐라는제주감귤500ML");
            put("90023", "동아오로나민C120ML");
            put("90024", "롯데칠성립톤밀크티240ML");
            put("90026", "웅진내사랑알로에180ML");
            put("90027", "빅토리아(베리베리)500ML");
            put("90029", "빙그레따옴백자몽포멜로235ML");
            put("90030", "동서)오션스프레이루비레드340ML");
            put("90031", "롯데)데일리C레몬1000C_(병)140ML");
            put("90032", "롯데)콜드오렌지250ML");
            put("90036", "매일)커피속에모카치노300ML");
            put("90037", "덴마크드링킹요구르트(플레인)310ML");
            put("90039", "가나초콜릿밀크300ML");
            put("90040", "해태)써니텐오렌지향250ML");
            put("90041", "동원샘물0.5L");
            put("90058", "롯데)2_부족할때아쿠아PET1.5L");
            put("90059", "롯데트로피카나스파클링사과1.5L");
            put("90061", "오이시)그린티포도");
            put("90062", "썬키스트후레쉬(포도)1.5L");
            put("90064", "파스퇴르)발렌시아오렌지1L");
            put("90065", "롯데델몬트콜드오렌지과즙100_");
            put("90066", "해태사브레105G");
            put("90067", "크라운)빅파이딸기216G");
            put("90068", "크라운츄러스84G");
            put("90071", "롯데)치토스매콤달콤한맛88G");
            put("90072", "오리온오징어땅콩98g");
            put("90073", "농심)고구마깡83g");
            put("90075", "오리온)오감자감자그라탕맛50G");
            put("90076", "크라운)죠리퐁74g");
            put("90086", "농심무파마큰사발112G");
            put("90087", "농심우육탕큰사발115G");
            put("90088", "농심육개장큰사발110G");
            put("90089", "롯데칸쵸컵88G");
            put("90090", "해태롤리폴리초코62G");
            put("90092", "동서)리츠샌드위치크래커초코77G");
            put("90093", "롯데)하비스트달콤고소100G");
            put("90094", "크라운국희땅콩샌드70G");
            put("90095", "롯데야채크래커83G");
            put("90106", "팔도)귀여운내친구뽀로로(블루베리)235ML");
            put("90107", "팔도)귀여운내친구뽀로로(사과)235ML");
            put("90108", "팔도)귀여운내친구뽀로로(바나나)235ML");
            put("90109", "팔도)귀여운내친구뽀로로(딸기)235ML");
            put("90110", "뽀로로샘물250ML");
            put("90111", "웅진초록매실180ML");
            put("90113", "팔도비락식혜238ML");
            put("90114", "동서맥심TOP마스터라떼275ML");
            put("90115", "동서맥심TOP스위트아메리카노275ML");
            put("90116", "롯데칸타타흑당라떼275ML");
            put("90117", "롯데핫식스더킹스톰");
            put("90118", "코카콜라)스프라이트250ML");
            put("90119", "레쓰비솔트커피타이완240ML");
            put("90120", "롯데마운틴듀330ML");
            put("90123", "해태허니버터칩60G");
            put("90124", "삼양)사또밥67G");
            put("90125", "농심양파링84g");
            put("90128", "오뚜기참깨라면용기110G");
            put("90130", "농심오징어짬뽕큰사발115G");
            put("90136", "오뚜기진짬뽕(큰컵)115G");
            put("90146", "롯데치토스스모키바베큐맛82G");
            put("90148", "오리온)더탱글마이구미100G");
            put("90186", "오리온땅콩강정80G");
            put("90187", "오리온)썬갈릭바게트맛64G");
            put("90222", "빙그레아카페라아메리카노");
            put("90223", "매일요구르트로어슈거");
            put("90225", "푸르밀가나초코우유");
            put("90226", "소와나무덴마크민트쵸코우유");
            put("90227", "소와나무덴마크드링킹요구르트딸기");
            put("90229", "오성물산티포버터쿠키");
            put("90230", "매일바이오드링킹요거트사과");
            put("90231", "롯데델몬트콜드비타민플러스포도100");
            put("90232", "서울우유아침에주스오렌지");
            put("90233", "빙그레아카페라사이즈업까페라떼");
            put("90234", "빙그레요구르트");
            put("90293", "프링글스또띠아나쵸치즈110G");
            put("A10024", "오리온)후레쉬베리");
            put("A20023", "동서식품)리츠크래커");
            put("A20024", "동서식품)오레오초콜릿크림");
            put("A20029", "크라운)뽀또");
            put("A30031", "(주)청우식품참깨스틱진");
            put("A30035", "농심김치큰사발면");
            put("A40029", "농심쌀국수");
            put("A40032", "로아커사로아커웨하스바닐라");
            put("A40033", "로아커사로아커웨하스샌드위치초콜릿");
        }};

        outData.rewind();
        outputMap.put(0, outData);
        Log.d("YoloV5Classifier", "mObjThresh: " + getObjThresh());

        Object[] inputArray = {imgData};
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
        byteBuffer.rewind();

        ArrayList<Recognition> detections = new ArrayList<Recognition>();

        float[][][] out = new float[1][output_box][numClass + 5];
        Log.d("YoloV5Classifier", "out[0] detect start");
        for (int i = 0; i < output_box; ++i) {
            for (int j = 0; j < numClass + 5; ++j) {
                if (isModelQuantized){
                    out[0][i][j] = oup_scale * (((int) byteBuffer.get() & 0xFF) - oup_zero_point);
                }
                else {
                    out[0][i][j] = byteBuffer.getFloat();
                }
            }
            // Denormalize xywh
            for (int j = 0; j < 4; ++j) {
                out[0][i][j] *= getInputSize();
            }
        }

        // 각 bounding box에 대해 가장 확률이 높은 Class 예측
        for (int i = 0; i < output_box; ++i){
            final int offset = 0;
            final float confidence = out[0][i][4];
            int detectedClass = -1;
            float maxClass = 0;

            final float[] classes = new float[labels.size()];
            for (int c = 0; c < labels.size(); ++c) {
                classes[c] = out[0][i][5 + c];  // classes: 각 class의 확률 계산
            }

            for (int c = 0; c < labels.size(); ++c) {
                if (classes[c] > maxClass) {
                    detectedClass = c;
                    maxClass = classes[c];
                }   // 가장 큰 확률의 class로 선정
            }

            final float confidenceInClass = maxClass * confidence;
            if (confidenceInClass > getObjThresh()) {
                final float xPos = out[0][i][0];
                final float yPos = out[0][i][1];

                final float w = out[0][i][2];
                final float h = out[0][i][3];
                Log.d("YoloV5Classifier",
                        Float.toString(xPos) + ',' + yPos + ',' + w + ',' + h);

                final RectF rect =
                        new RectF(
                                Math.max(0, xPos - w / 2),
                                Math.max(0, yPos - h / 2),
                                Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                detections.add(new Recognition("" + offset, nameMap.get(labels.get(detectedClass)),
                        confidenceInClass, rect, detectedClass));
            }
        }

        Log.d("YoloV5Classifier", "detect end");
        final ArrayList<Recognition> recognitions = nms(detections);
        return recognitions;
    }

    public boolean checkInvalidateBox(float x, float y, float width, float height, float oriW, float oriH, int intputSize) {
        // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        float halfHeight = height / 2.0f;
        float halfWidth = width / 2.0f;

        float[] pred_coor = new float[]{x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight};

        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        float resize_ratioW = 1.0f * intputSize / oriW;
        float resize_ratioH = 1.0f * intputSize / oriH;

        float resize_ratio = resize_ratioW > resize_ratioH ? resize_ratioH : resize_ratioW; //min

        float dw = (intputSize - resize_ratio * oriW) / 2;
        float dh = (intputSize - resize_ratio * oriH) / 2;

        pred_coor[0] = 1.0f * (pred_coor[0] - dw) / resize_ratio;
        pred_coor[2] = 1.0f * (pred_coor[2] - dw) / resize_ratio;

        pred_coor[1] = 1.0f * (pred_coor[1] - dh) / resize_ratio;
        pred_coor[3] = 1.0f * (pred_coor[3] - dh) / resize_ratio;

        // (3) clip some boxes those are out of range
        pred_coor[0] = pred_coor[0] > 0 ? pred_coor[0] : 0;
        pred_coor[1] = pred_coor[1] > 0 ? pred_coor[1] : 0;

        pred_coor[2] = pred_coor[2] < (oriW - 1) ? pred_coor[2] : (oriW - 1);
        pred_coor[3] = pred_coor[3] < (oriH - 1) ? pred_coor[3] : (oriH - 1);

        if ((pred_coor[0] > pred_coor[2]) || (pred_coor[1] > pred_coor[3])) {
            pred_coor[0] = 0;
            pred_coor[1] = 0;
            pred_coor[2] = 0;
            pred_coor[3] = 0;
        }

        // (4) discard some invalid boxes
        float temp1 = pred_coor[2] - pred_coor[0];
        float temp2 = pred_coor[3] - pred_coor[1];
        float temp = temp1 * temp2;
        if (temp < 0) {
            Log.e("checkInvalidateBox", "temp < 0");
            return false;
        }
        if (Math.sqrt(temp) > Float.MAX_VALUE) {
            Log.e("checkInvalidateBox", "temp max");
            return false;
        }

        return true;
    }
}
