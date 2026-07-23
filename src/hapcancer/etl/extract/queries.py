# ------- QUERIES --------


# ------------------------
   

# -- wrapper for specific queries
QUERY_HELPER = {
    "biopsy": {
        "counter": Q_BIOPSY_RAW_COUNTER,
        "content": Q_BIOPSY_RAW
    },
    "anamnesis": {
        "V1": {
            "content": Q_ANAMNESIS_PAGINATED,
            "counter": Q_ANAMNESIS_COUNTER
        },
        "V2": {
            "content": Q_ANAMNESIS_PAGINATED,
            "counter": Q_ANAMNESIS_COUNTER
        }
    },
    "person": {
        "V1": { "content": Q_PESSOA_FROM_PROCEDURES_PAGINATED },
        "V2": { "content": Q_PESSOA_FROM_PROCEDURES_PAGINATED }
    },
    "patient": {
        "V1": { "content": Q_PACIENTE_FROM_PROCEDURES_PAGINATED },
        "V2": { "content": Q_PACIENTE_FROM_PROCEDURES_PAGINATED }
    },
    "user": {
        "V1": { "content": Q_USER_FROM_PROCEDURES_PAGINATED },
        "V2": { "content": Q_USER_FROM_PROCEDURES_PAGINATED }
    },
    "mammogram": {
        "V1": {
            "content": Q_LOAD_MAMMOGRAM_EXAMS_BY_PROCEDURE_PAGINATED,
            "counter": Q_LOAD_MAMMOGRAM_EXAMS_BY_PROCEDURE_COUNTER
        },
        "V2": {
            "content": Q_LOAD_MAMMOGRAM_EXAMS_BY_PROCEDURE_PAGINATED,
            "counter": Q_LOAD_MAMMOGRAM_EXAMS_BY_PROCEDURE_COUNTER
        }
    }
}

