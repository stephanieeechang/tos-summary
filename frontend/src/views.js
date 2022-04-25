import React, {useEffect, useState} from "react";
import {baseUrlExtractiveSummaryServerRemote as baseUrl} from "./constants";

import ReactMarkdown from "react-markdown";

const urlExtractiveSummarySummarize = `${baseUrl}/summarize`
const fetchHeader = {
    method: "get",
    headers: {
        Accept: "text/plain",
        "Content-Type": "text/plain;charset=UTF-8"
    }
}

export function ExtractiveSummaryDemoView(props) {

    const availableDocTypes = ["privacy policy", "terms of service"]

    const [privacyPolicyDocuments, setPrivacyPolicyDocuments] = useState([]);
    const [privacyPolicyDocumentsReady, setPrivacyPolicyDocumentsReady] = useState(false);


    // controlled inputs
    const [whichForm, setWhichForm] = useState("select")
    const [valueInputDocType, setValueInputDocType] = useState("privacy policy")
    const [valueInputDocName, setValueInputDocName] = useState("")
    const [valueInputCustomText, setValueInputCustomText] = useState("")

    // outputs
    const decoderUTF8 = new TextDecoder('utf-8')
    const [outputRequested, setOutputRequested] = useState(false)
    const [valueOutput, setValueOutput] = useState("")
    const [isLoadingOutput, setIsLoadingOutput] = useState(false)

    function consumeResponseStream(res: Response) {
        const reader = res.body.getReader()
        // let chunkCount = 0;
        let readText = "";
        function consumeChunk({done, value}) {
            if (!isLoadingOutput) setIsLoadingOutput(true)
                if (done) {
                    // console.log(`Done with chunk count ${chunkCount}`)
                    // console.log(readText)
                    setIsLoadingOutput(false)
                    // console.log(`readText: ${readText}`)
                    // console.log(`state: ${valueOutput}`)
                } else {
                    // console.log(`consumeChunk: received chunk ${chunkCount++}`)
                    const decoded = decoderUTF8.decode(value, {stream: true})
                    // console.log(decoded)
                    readText = readText.concat(` ${decoded}`)
                    // console.log(readText)
                    setValueOutput(readText)
                    return reader.read().then(consumeChunk, err => console.log(err))
                }
            }
        reader.read().then(({done, value}) => consumeChunk({done, value}), err => console.log(err))
    }

    // form
    function handleSubmit(event: SubmitEvent) {
        event.preventDefault()
        setValueOutput("")
        setOutputRequested(true)
        setIsLoadingOutput(true)
        console.log(event.target.id)
        if (event.target.id === "form-select") {
            setWhichForm("select")
            fetch(
                `${urlExtractiveSummarySummarize}?docType=${valueInputDocType}&docName=${valueInputDocName}`,
                fetchHeader
            ).then(res => consumeResponseStream(res), err => console.error(err))
        }
        else
            if (event.target.id === "form-custom") {
                setWhichForm("custom")
                fetch(
                    urlExtractiveSummarySummarize,
                    {
                        method: "post",
                        headers: {
                            Accept: "application/json",
                            "Content-Type": "application/json;charset=UTF-8"
                        },
                        body: JSON.stringify({"custom_text": valueInputCustomText})
                    }
                ).then(res => consumeResponseStream(res), err => console.error(err))
                    // .then(res => console.log(res.text().then(t => console.log(t))))
            }
        }

        // fetch the list of privacy policy documents
        useEffect(() => {
            // fetch a list of available privacy policy documents
            fetch(
                `${baseUrl}/privacy_policies`,
                {
                    method: "GET",
                    headers: {
                        Accept: "application/json",
                        "Content-Type": "application/json;charset=UTF-8"
                    }
                }
            ).then(res => {
                res.json().then(docs => {
                    setPrivacyPolicyDocuments(docs)
                    setPrivacyPolicyDocumentsReady(true)
                })
            })
        }, [])


        return (<div className="container-fluid m-0 p-0">
            <header className="container-fluid p-5 bg-primary text-white text-center">
                <h1 className="fw-light">Extractive Summary of Privacy Policies and Terms of Service</h1>
                <p className="fs-5">using task-specific pretrained LegalBERT</p>
            </header>
            <main className="container m-auto">
                {/* forms */}
                <div className="row mx-auto my-5 justify-content-center align-content-center">
                    <div className="col-12 mb-3">
                        <h2 className="fw-light text-center">See an example</h2>
                    </div>
                    <div className="col mb-3">
                        <form id="form-select" action={urlExtractiveSummarySummarize} method="get" target="_self"
                              onSubmit={handleSubmit}>
                            <label htmlFor="select-doc-type" className="input-group-text" hidden>Summarize
                                Example</label>
                            <label htmlFor="input-select-doc-name" className="input-group-text" hidden>of</label>
                            <div className="input-group m-auto pe-auto">
                                <span className="input-group-text">Summarize the</span>
                                <select className="form-select" name="docType" id="select-doc-type" required
                                        onChange={event => setValueInputDocType(event.target.value)}>
                                    {availableDocTypes.map(docType => (<option value={docType}>{docType}</option>))}
                                </select>
                                <span className="input-group-text">of</span>
                                <input className="form-control" type="text" name="docName" required
                                       onChange={event => setValueInputDocName(event.target.value)} list="list-doc-name"/>
                                <datalist id="list-doc-name">
                                    {privacyPolicyDocuments.map(documentName => (
                                        <option value={documentName}>{documentName}</option>))}
                                </datalist>
                                <button className="btn btn-primary" disabled={isLoadingOutput} type="submit">
                                    {
                                        isLoadingOutput &&
                                        <span className="spinner-border spinner-border-sm me-2 px-1" role="status"
                                              aria-hidden="true" />
                                    }
                                    Summarize
                                </button>
                            </div>
                        </form>
                    </div>
                    <div className="col-12 mb-1"><p className="fw-bold fs-5 text-center m-0 p-0">or</p></div>
                    <div className="col-12 mb-3"><h2 className="fw-light text-center">Summarize your own text</h2></div>
                    <div className="col mb-3">
                        <form id="form-custom" action={urlExtractiveSummarySummarize} method="get" target="_self" onSubmit={handleSubmit}>
                            <label htmlFor="input-custom-text" hidden>Custom Text</label>
                            <div className="input-group m-auto pe-auto">
                                <textarea className="form-control" name="input-custom-text" id="input-custom-text"
                                          required cols="30" rows="10"
                                          placeholder="Lorem ipsum dolor sit amet, consectetur adipiscing elit..."
                                          onChange={event => setValueInputCustomText(event.target.value)}/>
                                <button className="btn btn-primary" disabled={isLoadingOutput} type="submit">
                                    {
                                        isLoadingOutput &&
                                        <span className="spinner-border spinner-border-sm me-2 px-1" role="status"
                                              aria-hidden="true" />
                                    }
                                    Summarize
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
                {/* output */}
                {
                    outputRequested &&
                    <div className="row mx-auto my-5 justify-content-center align-content-center">
                        <div className="col-12 mb-3">
                            <h2 className='fw-light text-center'>Summary</h2>
                        </div>
                        {
                            valueOutput.length >= 0 &&
                            <div className="col p-3 bg-light border-3">
                                <ReactMarkdown>
                                    {valueOutput}
                                </ReactMarkdown>
                                {
                                    (isLoadingOutput && outputRequested) &&
                                    <span className="spinner-border spinner-border-sm mx-3">
                                        <span className="visually-hidden">Loading</span>
                                    </span>
                                }
                            </div>
                        }
                    </div>
                }
            </main>
        </div>)
    }