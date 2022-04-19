import React, {useEffect, useState} from "react";
import {baseUrlExtractiveSummaryServerLocalHost} from "./constants";

export function ExtractiveSummaryDemoView(props) {

    const availableDocTypes = ["privacy policy", "terms of service"]

    const [privacyPolicyDocuments, setPrivacyPolicyDocuments] = useState([]);
    const [privacyPolicyDocumentsReady, setPrivacyPolicyDocumentsReady] = useState(false);

    useEffect(() => {
        // fetch a list of available privacy policy documents
        fetch(
            `${baseUrlExtractiveSummaryServerLocalHost}/privacy_policies`,
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
        <main className="container-fluid">
            <div className="row mx-auto my-5 mx-auto justify-content-center align-content-center">
                <div className="col">
                    <div className="input-group m-auto pe-auto">
                        <label htmlFor="select-doc-type" className="input-group-text">Summarize</label>
                        <select name="select-doc-type" id="select-doc-type">
                            {availableDocTypes.map(docType => (<option value={docType}>{docType}</option>))}
                        </select>
                        <label htmlFor="input-select-doc-name" className="input-group-text">of</label>
                        <input list="list-doc-name" placeholder={privacyPolicyDocuments[0]}/>
                        <datalist id="list-doc-name">
                            {privacyPolicyDocuments.map(documentName => (<option value={documentName}>{documentName}</option>))}
                        </datalist>
                    </div>
                </div>
            </div>
        </main>
    </div>)
}