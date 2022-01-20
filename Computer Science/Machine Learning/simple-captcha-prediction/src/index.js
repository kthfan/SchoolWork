


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
	
	var model = null;
	Promise.all([
		load_js("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js")
	]).then(() => {
		model = build_model();
		loadWeights(model, WEIGHTS);
		
		document.getElementsByClassName("js-captcha")[0].onload = function(){
			refreshCaptcha();
		}
		
		setTimeout(refreshCaptcha, 1);
	});
	
	async function refreshCaptcha(){
		tf.engine().startScope();
		try{
			captchaElem.value = predict(model, document.getElementsByClassName("js-captcha")[0]);
			if(checkCapthaError()) submitBn.click();
		}catch(e){
			console.error(e);
			// setTimeout(refreshCaptcha, 500);
		}
		tf.engine().endScope();
	}
	
	function checkCapthaError(){
		return document.getElementsByClassName("help-block").length ==1 &&
			Array.from(document.getElementsByClassName("help-block")).some(elem=>{
			  if(elem.childNodes[0] && elem.childNodes[0].childNodes[1]){
				elem = elem.childNodes[0].childNodes[1];
				return elem.textContent.indexOf("驗證碼") !== -1;
			  }
			  return false;
			})
	}
}



