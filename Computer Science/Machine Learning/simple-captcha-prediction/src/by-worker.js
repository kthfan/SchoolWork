

const workerBlob = new Blob([`

importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js");

const WEIGHTS = [[[[[0.5738179683685303]], [[0.9089720845222473]], [[1.878428339958191]], [[1.5401345491409302]], [[0.4327313005924225]]], [[[1.1174379587173462]], [[0.5479282736778259]], [[1.86570143699646]], [[1.3090124130249023]], [[1.1479483842849731]]], [[[-0.5226042866706848]], [[-0.09183017164468765]], [[-0.41744205355644226]], [[0.10690238326787949]], [[0.16504114866256714]]], [[[-0.7289248108863831]], [[-0.47181636095046997]], [[-1.0635185241699219]], [[-0.2546493709087372]], [[0.12296561896800995]]], [[[-0.8994237780570984]], [[-0.45175012946128845]], [[-0.7355265617370605]], [[-0.4982589781284332]], [[0.015483151189982891]]]], [-0.3260951638221741], [[-0.5146745443344116, 0.01716594211757183, -0.09422832727432251, 0.4746061861515045], [-0.11935438960790634, 1.5017422437667847, -0.2092299908399582, -0.3529056906700134], [0.2861678898334503, 0.018816182389855385, 0.11925645172595978, -0.16657836735248566], [0.43444427847862244, 1.3855096101760864, -0.3908580541610718, -0.19503438472747803], [0.3818058669567108, 1.198350191116333, 0.4419322907924652, 0.718563973903656], [-0.05207217484712601, 0.06561774760484695, 0.2608788013458252, 0.5486858487129211], [-0.37405896186828613, 0.0015566007932648063, 0.2981886565685272, 0.2371513396501541], [-0.4211955666542053, -0.22480811178684235, 0.34574997425079346, -1.19247567653656], [0.26376771926879883, -0.3654508590698242, 0.21079538762569427, -0.6711538434028625], [-0.32007232308387756, 0.04502009600400925, -0.30607980489730835, -0.3887516260147095], [0.09283497184515, 1.6174993515014648, -0.3743370473384857, 0.2440575808286667], [-0.7321068644523621, -0.09578262269496918, -0.18029381334781647, -0.22045356035232544], [-0.07044795155525208, -0.007807496003806591, -0.38963204622268677, -0.25867828726768494], [-0.0670156478881836, 0.03858484327793121, -0.5568774938583374, -0.35166698694229126], [0.9052507877349854, 0.30861136317253113, -0.4552311897277832, 0.02214280515909195], [0.6467926502227783, -0.14132849872112274, 0.1144900768995285, 0.09752432256937027], [-0.2573392391204834, 0.0657094269990921, -0.43401190638542175, -0.5329514741897583], [-0.08384643495082855, 0.34002944827079773, -0.4024886190891266, 0.49981656670570374], [-0.5899662971496582, 0.10857801139354706, -0.5770108699798584, -0.18662121891975403], [0.2396615594625473, 0.6148038506507874, -0.568463921546936, 0.6558638215065002], [0.3561047613620758, 0.1899968534708023, -0.03118877299129963, -0.09990986436605453], [-0.24837474524974823, -0.19849130511283875, -0.04915526509284973, 0.3995685279369354], [0.025243932381272316, 0.7142778635025024, -0.1313670128583908, -0.27171266078948975], [-0.33027368783950806, 0.3558429777622223, -0.3659612834453583, 0.2963963747024536], [-0.18952225148677826, 0.39697399735450745, -0.09709574282169342, -0.29745104908943176], [-0.4920235574245453, 0.10251868516206741, 0.697374165058136, 0.6785368323326111], [0.00925495382398367, -0.11712668836116791, 0.5187991261482239, 0.3195819556713104], [0.3235263526439667, -0.28163865208625793, 0.414529412984848, 0.27762115001678467], [-0.6428472995758057, -0.32381170988082886, 0.4653394818305969, 0.2856827974319458], [-0.4524711072444916, 0.17642463743686676, -0.25288817286491394, 0.5692474246025085], [-0.38372138142585754, 0.31432098150253296, 0.3214988112449646, 0.33684229850769043], [-0.7933276295661926, 0.4972286522388458, 0.1710566282272339, 0.000915799755603075], [-0.28241321444511414, 0.3650517165660858, -0.3794131875038147, -0.09754545241594315], [-0.2541164457798004, -0.45822009444236755, -0.5003936290740967, 0.602577269077301], [-0.5141561627388, -0.2951104938983917, -0.014458790421485901, 0.46557649970054626], [-0.46767503023147583, 0.09558743238449097, -0.13885799050331116, 0.15488819777965546]], [0.2932751476764679, -0.07228467613458633, 0.1090247854590416, -0.09220453351736069], [[0.3620151877403259, 0.9746187925338745, -0.17451851069927216, -0.30864259600639343, 0.8385673761367798, 0.35575905442237854, -0.23213782906532288, 0.03020618110895157, -0.32581037282943726, 0.7703232765197754], [-0.32313716411590576, -0.9823414087295532, -0.6907592415809631, -1.0066243410110474, -0.9131797552108765, 0.543189287185669, 0.275577187538147, 0.4813116788864136, -0.7084596157073975, 0.273355096578598], [-0.010843872092664242, -0.8485767841339111, -0.9020207524299622, 0.8979983925819397, 0.6996585726737976, -0.8502582311630249, -0.10162859410047531, 0.6614603400230408, -0.5267243385314941, -0.9703415632247925], [-0.2749040722846985, -0.46851474046707153, -0.7339933514595032, -0.35980290174484253, 0.38613206148147583, 0.33924272656440735, 0.5041974186897278, -0.7207197546958923, 0.3532043993473053, -0.7357217073440552]], [-2.046898365020752, -0.06522023677825928, -0.4040914475917816, -0.2085680216550827, -0.34100109338760376, 0.1047598347067833, 0.13982363045215607, -0.1796296089887619, 0.6376714706420898, -0.0676136463880539]];

function load_js(src){
	let script = document.createElement("script");
	script.src = src;
	script.async = true;
	document.body.appendChild(script);
	let [ok, err] = [null, null]; 
	let promise = new Promise((resolve, reject) => {
		ok = resolve;
		err = reject;
	});
	// let timdId = setTimeout(()=>ok(), 5000); // wait for 5s.
	script.onload = () => {
		//clearTimeout(timdId);
		ok();
		
	}
	return promise;
}

function convertByColor(I){
	I = I.neg().add(255);
	I = tf.mean(I, 2);
	I = tf.cast(I.greater(160), "float32");
    return I;
}

function splitImage(I){
	
	I = tf.slice(I, [3, 11], [20, 79]);
	let [h, w] = I.shape;
	let w2 = Number.parseInt(w/6);
	let result = Array.from(Array(6))
		.map((_, i) => tf.slice(I, [0, i*w2], [I.shape[0], w2]))
		.map(img => tf.pad(img, [[4,4],[7,8]]));
	
	return result
}
function imagesToTensors(image_list){
	var result = tf.concat(image_list, 0);
	result = result.reshape([image_list.length, 28, 28, 1]);
	return result;
}
function loadWeights(model, weights){
	weights = weights.map(w => tf.tensor(w));
	model.setWeights(weights);
}
function build_model(){
	let inputs = tf.input({shape:[28, 28, 1], name:'inputs'});
	
	let conv1 = tf.layers.conv2d({filters :1, kernelSize: 5, name:'conv1'}).apply(inputs);
	let pool1 = tf.layers.maxPool2d({poolSize :[4, 4], name:'pool1'}).apply(conv1);
	let flatten = tf.layers.flatten({name: "flatten"}).apply(pool1);
	
	let dense1 = tf.layers.dense({units: 4, name: "dense1"}).apply(flatten)
	
	let drop1 = tf.layers.dropout({rate: 0.2, name:"drop1"}).apply(dense1);
	let outputs = tf.layers.dense({units: 10, activation: "softmax", name:"outputs"}).apply(drop1);

	let model = tf.model({inputs: inputs, outputs: outputs});
	
	// model.compile({optimizer: 'adam',loss: 'categoricalCrossentropy',metrics: ['accuracy']})
	// model.summary()
	return model;
}

function predict(model, I){
	//let I = tf.browser.fromPixels(imgElem);
	
	I = convertByColor(I);
	let images = splitImage(I);
	let tensors = imagesToTensors(images);
	let pY = model.predict(tensors);
	return pY.argMax(1).arraySync().join("");
}

var model = build_model();
loadWeights(model, WEIGHTS);

onmessage = function(e) {
	tf.engine().startScope();
	let I = tf.tensor(new Uint8Array(e.data.buffer), [28, 100, 4]);
	I = tf.slice(I, [0, 0, 0], [28, 100, 3]);
	let result = predict(model, I);
	tf.engine().endScope();
	postMessage(result);
}
`]);

if(document.readyState === "complete" || document.readyState === "interactive") {
	setTimeout(main, 1);
}else{
	document.addEventListener("DOMContentLoaded", main);
}

async function main(){
	const captchaElem = document.getElementsByClassName("fs-form-captcha-field")[0];
	const refreshBn = document.getElementsByClassName("js-refresh-captcha")[0];
	const submitBn = document.getElementsByClassName("btn btn-primary btn-lg btn-block")[0];
	
	if(!captchaElem) return;
	
	var worker = new Worker(window.URL.createObjectURL(workerBlob));
	
	worker.onmessage  = e =>{
		captchaElem.value = e.data;
		if(checkCapthaError()) submitBn.click();
	}
	
	document.getElementsByClassName("js-captcha")[0].addEventListener("load", function(){
		refreshCaptcha();
	});
	setTimeout(refreshCaptcha, 1);
	
	function refreshCaptcha(){
		let I = document.getElementsByClassName("js-captcha")[0];
		let canvas = document.createElement("canvas");
		canvas.height = I.naturalHeight;
		canvas.width = I.naturalWidth;
		let ctx = canvas.getContext("2d");

		ctx.drawImage(I, 0, 0);
		let data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
		worker.postMessage(data);
		
	}
	
	function checkCapthaError(){
		return document.getElementsByClassName("help-block").length ==1 &&
			Array.from(document.getElementsByClassName("help-block")).some(elem=>{
			  if(elem.childNodes[0] && elem.childNodes[0].childNodes[1]){
				elem = elem.childNodes[0].childNodes[1];
				return elem.textContent.indexOf("驗證") !== -1 ||
					elem.textContent.indexOf("验证") !== -1 ||
					elem.textContent.indexOf("aptcha") !== -1 ;
			  }
			  return false;
			})
	}
}



