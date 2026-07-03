const documentList = document.querySelector("#documentList");
const uploadForm = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const askForm = document.querySelector("#askForm");
const questionInput = document.querySelector("#questionInput");
const messages = document.querySelector("#messages");
const rebuildButton = document.querySelector("#rebuildButton");

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function addMessage(text, role) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  messages.appendChild(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

async function loadDocuments() {
  const response = await fetch("/api/documents");
  const data = await response.json();
  documentList.innerHTML = "";

  if (!data.files.length) {
    const empty = document.createElement("div");
    empty.className = "status";
    empty.textContent = "Nessun documento caricato.";
    documentList.appendChild(empty);
    return;
  }

  for (const file of data.files) {
    const row = document.createElement("div");
    row.className = "document-row";

    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "document-name";
    name.textContent = file.name;

    const meta = document.createElement("div");
    meta.className = "document-meta";
    meta.textContent = formatBytes(file.size);

    const badge = document.createElement("span");
    badge.className = file.indexed ? "badge" : "badge warning";
    badge.textContent = file.indexed ? "indicizzato" : "da sincronizzare";

    info.appendChild(name);
    info.appendChild(meta);
    info.appendChild(badge);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Elimina";
    deleteButton.addEventListener("click", async () => {
      deleteButton.disabled = true;
      await fetch(`/api/documents/${encodeURIComponent(file.name)}`, {
        method: "DELETE",
      });
      await loadDocuments();
    });

    row.appendChild(info);
    row.appendChild(deleteButton);
    documentList.appendChild(row);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const button = uploadForm.querySelector("button");
  button.disabled = true;
  button.textContent = "Indicizzazione...";

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Upload fallito.");
    }

    fileInput.value = "";
    await loadDocuments();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Carica e indicizza";
  }
});

rebuildButton.addEventListener("click", async () => {
  rebuildButton.disabled = true;
  rebuildButton.textContent = "Sincronizzazione...";

  try {
    const response = await fetch("/api/rebuild", { method: "POST" });
    if (!response.ok) throw new Error("Sincronizzazione fallita.");
    await loadDocuments();
  } catch (error) {
    alert(error.message);
  } finally {
    rebuildButton.disabled = false;
    rebuildButton.textContent = "Sincronizza tutto";
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  addMessage(question, "user");
  const answer = addMessage("", "assistant");

  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    const error = await response.json();
    answer.textContent = error.detail || "Errore durante la risposta.";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    answer.textContent += decoder.decode(value, { stream: true });
    messages.scrollTop = messages.scrollHeight;
  }
});

loadDocuments();
